#!/usr/bin/env python3
"""
ADP trajectory -> prior experience memory converter.

This is a deterministic extractor, not a summarizer. It converts ADP JSONL
trajectories into:
  1. machine-readable memory JSONL, one memory record per trajectory;
  2. concise rendered .txt files suitable for SWE-Agent/OpenHands injection.

No LLM calls, embeddings, network access, random IDs, or filesystem-dependent
path resolution are used. Same input + same extraction flags produces
byte-identical output.

All action_index / observation_index values are zero-based positions in the
ADP trajectory content list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

MEMORY_SCHEMA_VERSION = "1.2.0"
SCRIPT_NAME = "adp_to_memory.py"

INSPECT_TOOLS = frozenset({"Read", "Grep", "Glob"})
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit"})
SEARCH_ANCHOR_TOOLS = frozenset({"Grep", "Glob"})
PLANNING_TOOLS = frozenset({"TodoWrite"})

SOURCE_EXTENSIONS = frozenset(
    {
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs",
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cs", ".rb", ".php",
        ".swift", ".kt", ".kts", ".scala", ".sh", ".bash", ".zsh", ".sql",
        ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json", ".xml", ".html",
        ".css", ".scss", ".md", ".rst",
    }
)
TEST_EXTENSIONS = SOURCE_EXTENSIONS

# Deterministic command rules. First matching class wins in classify_command().
_CMD_DESTRUCTIVE = re.compile(
    r"(?is)(^|[;&|]\s*)(sudo\s+)?rm\s+(-[A-Za-z]*[rf][A-Za-z]*|-r\s+-f|-f\s+-r)\b|"
    r"\bgit\s+clean\s+-[A-Za-z]*[xdf][A-Za-z]*\b|"
    r"\bgit\s+reset\s+--hard\b|"
    r"\bfind\b[^\n;]*\s-delete\b"
)
_CMD_SETUP = re.compile(
    r"(?is)\b(git\s+clone|git\s+-C\s+\S+\s+fetch|git\s+fetch|git\s+checkout|"
    r"git\s+-C\s+\S+\s+checkout|git\s+submodule|conda\s+create|python\s+-m\s+venv|"
    r"virtualenv|apt-get|brew\s+install|mkdir\s+-p|cp\s+-R)\b"
)
_CMD_TEST = re.compile(
    r"(?is)(^|[;&|]\s*)(python\s+-m\s+pytest|pytest|py\.test|tox|nosetests|"
    r"python\s+-m\s+unittest|unittest|npm\s+(run\s+)?test|pnpm\s+(run\s+)?test|"
    r"yarn\s+(run\s+)?test|go\s+test|cargo\s+test|mvn\s+test|gradle\s+test|"
    r"./gradlew\s+test|rspec|bundle\s+exec\s+rspec|python3?\s+[^\n;]*test[^\n;]*\.py|python\s+[^\n;]*test[^\n;]*\.py)\b"
)
_CMD_BUILD_INSTALL = re.compile(
    r"(?is)\b(pip3?\s+install|python3?\s+-m\s+pip\s+install|uv\s+pip\s+install|"
    r"poetry\s+install|pipenv\s+install|npm\s+install|npm\s+ci|pnpm\s+install|"
    r"yarn\s+install|cargo\s+build|cargo\s+check|npm\s+run\s+build|pnpm\s+build|"
    r"yarn\s+build|mvn\s+(install|package|compile)|gradle\s+build|./gradlew\s+build|"
    r"make(\s|$)|python3?\s+setup\.py\s+(build|install|develop|build_ext))\b"
)
_CMD_DIAGNOSTIC = re.compile(
    r"(?is)^\s*(cd\s+\S+\s*&&\s*)?(ls|find|grep|rg|sed|cat|head|tail|wc|pwd|"
    r"git\s+status|git\s+diff|git\s+log|git\s+show|git\s+grep|python\s+-c)\b"
)

_HEREDOC = re.compile(r"<<\s*['\"]?[A-Za-z0-9_:-]+['\"]?")

_FAILURE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^FAILED\b"), "pytest_failure"),
    (re.compile(r"^ERROR\b"), "error_line"),
    (re.compile(r"^Traceback\b"), "traceback"),
    (re.compile(r"^AssertionError\b"), "assertion_error"),
    (re.compile(r"^Exception\b"), "exception"),
    (re.compile(r"^RuntimeError\b"), "runtime_error"),
    (re.compile(r"^TypeError\b"), "type_error"),
    (re.compile(r"^ValueError\b"), "value_error"),
    (re.compile(r"^ImportError\b"), "import_error"),
    (re.compile(r"^ModuleNotFoundError\b"), "module_not_found"),
    (re.compile(r"^panic:"), "panic"),
    (re.compile(r"^FAIL:"), "go_test_failure"),
    (re.compile(r"^FAIL\b"), "go_test_failure"),
    (re.compile(r"^E\s+"), "pytest_error_line"),
]

_FAILURE_BUCKETS = (
    "test_failures",
    "runtime_failures",
    "setup_failures",
    "install_failures",
    "other_failures",
)

COMMAND_BUCKETS = (
    "test_commands",
    "diagnostic_commands",
    "setup_commands",
    "destructive_commands",
    "build_install_commands",
    "other_commands",
)

EDIT_BUCKETS = (
    "source_edits",
    "test_edits",
    "scratch_edits",
    "submission_artifacts",
    "other_edits",
)

_TEMPLATE_HEADERS = {
    "generic": (
        "This is deterministic evidence from a previous trajectory on a related issue.\n"
        "Do not copy the old patch blindly. Verify everything in the current repository."
    ),
    "swe-agent": (
        "This is read-only prior trajectory evidence for SWE-Agent.\n"
        "Do not apply the old patch verbatim. Re-derive the fix in the current repository."
    ),
    "openhands": (
        "This is read-only prior trajectory evidence for OpenHands.\n"
        "Use it as context only; verify all behavior in the current workspace."
    ),
}

INJECTION_WARNING = (
    "This memory is from a prior related issue. Use it as evidence only. "
    "Do not copy patches blindly."
)


# ---------------------------------------------------------------------------
# Normalization and hashing
# ---------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    """Normalize text deterministically: LF line endings, trailing whitespace removed."""
    if value is None:
        s = ""
    elif isinstance(value, str):
        s = value
    else:
        s = str(value)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n"))


SYSTEM_REMINDER_RE = re.compile(r"(?is)\n?\s*<system-reminder>.*?</system-reminder>\s*\n?")
DIFF_FENCE_RE = re.compile(r"(?is)```diff\s*\n.*?\n```")
HARD_FORBIDDEN_TOKENS_RE = re.compile(
    r"(?is)(diff --git|```diff|manual\.ya?ml|agent-summary:|agent-session-paths:|\.realign/sessions/|"
    r"model_patch|_preds\.json|prediction\.json|<system-reminder>|</system-reminder>|reasoning_content|todowrite)"
)
HARNESS_PROMPT_JUNK_RE = re.compile(
    r"(?is)(manual\.ya?ml|agent-summary:|agent-session-paths:|\.realign/sessions/|"
    r"fix this bug to solve the issue based on manual|whenever you read a file)"
)
UNIFIED_DIFF_HEADER_RE = re.compile(
    r"(?i)^(?:diff --git\b|index\s+[0-9a-f]+\.\.[0-9a-f]+\b|---\s+[ai]/|\+\+\+\s+[bw]/|@@(?:\s|$))"
)
PATCH_WRITE_RE = re.compile(r"""(?is)\b(?:cat|tee)\s*(?:>>?|>\|?)\s*(?:'([^']+)'|"([^"]+)"|([^\s;|&<>]+))""")


def strip_system_reminders(text: Any) -> str:
    cleaned = SYSTEM_REMINDER_RE.sub("\n", normalize_text(text))
    return normalize_text(cleaned)


def strip_diff_blocks(text: Any) -> str:
    cleaned = DIFF_FENCE_RE.sub("\n", normalize_text(text))
    out_lines: list[str] = []
    in_unified_diff = False
    for raw_line in cleaned.split("\n"):
        line = raw_line.rstrip()
        stripped = line.lstrip()
        if in_unified_diff:
            if not stripped or line.startswith((" ", "\t", "+", "-", "\\")):
                continue
            if UNIFIED_DIFF_HEADER_RE.match(stripped):
                continue
            in_unified_diff = False
        if UNIFIED_DIFF_HEADER_RE.match(stripped):
            in_unified_diff = True
            continue
        out_lines.append(line)
    return normalize_text("\n".join(out_lines))


def strip_harness_prompt_junk(text: Any) -> str:
    cleaned = strip_system_reminders(strip_diff_blocks(text))
    out_lines: list[str] = []
    for raw_line in cleaned.split("\n"):
        if HARNESS_PROMPT_JUNK_RE.search(raw_line):
            continue
        out_lines.append(raw_line)
    return normalize_text("\n".join(out_lines))


def strip_harness_prompt_junk_from_value(value: Any) -> Any:
    if isinstance(value, str):
        return strip_harness_prompt_junk(value)
    if isinstance(value, list):
        return [strip_harness_prompt_junk_from_value(v) for v in value]
    if isinstance(value, tuple):
        return [strip_harness_prompt_junk_from_value(v) for v in value]
    if isinstance(value, dict):
        return {k: strip_harness_prompt_junk_from_value(v) for k, v in value.items()}
    return value


def text_contains_hard_forbidden_payload(text: Any) -> bool:
    return bool(HARD_FORBIDDEN_TOKENS_RE.search(normalize_text(text)))


def original_content_index(item: dict[str, Any], fallback: int) -> int | None:
    strip_source = item.get("strip_source") if isinstance(item.get("strip_source"), dict) else {}
    value = strip_source.get("original_adp_content_index")
    return value if isinstance(value, int) else None


def add_original_index(entry: dict[str, Any], key: str, item: dict[str, Any], fallback: int) -> None:
    original = original_content_index(item, fallback)
    if original is not None:
        entry[key] = original


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Return deterministic capped text. max_chars==0 means keep no chars."""
    if max_chars < 0:
        raise ValueError("max_chars must be non-negative")
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def canonicalize_json_value(value: Any) -> Any:
    """Canonicalize JSON-like values for hashing; normalize all strings."""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [canonicalize_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [canonicalize_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): canonicalize_json_value(value[k]) for k in sorted(value.keys(), key=str)}
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonicalize_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8", errors="replace")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stable_json_dumps(value: Any) -> str:
    """Stable JSONL serialization for output records."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def short_description(value: Any, max_chars: int = 240) -> str:
    text = strip_harness_prompt_junk(value)
    text, _ = truncate_text(text, max_chars)
    return text


# ---------------------------------------------------------------------------
# Path normalization and classification
# ---------------------------------------------------------------------------


def normalize_path_with_raw(path_value: Any) -> tuple[str, str]:
    """Return (repo_relative_path, raw_path) without inspecting the filesystem."""
    raw = normalize_text(path_value).strip()
    p = raw.replace("\\", "/").strip()
    p = re.sub(r"/+", "/", p)
    while p.startswith("./"):
        p = p[2:]

    # /tmp/.../testbed/foo.py -> foo.py, /private/tmp/.../testbed/foo.py -> foo.py
    m_abs = re.search(r"(?:^|/)(?:testbed)/(.*)$", p)
    if m_abs and (p.startswith("/") or "/tmp/" in p or p.startswith("tmp/")):
        p = m_abs.group(1)

    # swebench_9_15/testbed/foo.py -> foo.py
    p = re.sub(r"^(?:[^/]*swebench[^/]*/)+testbed/", "", p)
    p = re.sub(r"^swebench_[^/]+/testbed/", "", p)

    # testbed/foo.py -> foo.py
    if p.startswith("testbed/"):
        p = p[len("testbed/") :]

    # A second pass handles ./testbed after slash collapse/strip.
    while p.startswith("./"):
        p = p[2:]
    p = re.sub(r"/+", "/", p)
    return p, raw


def normalize_path(path_value: Any) -> str:
    return normalize_path_with_raw(path_value)[0]


def is_workspace_root_path(path: str) -> bool:
    p = normalize_path(path).strip().lower()
    return p in {"", ".", "./", "testbed", "./testbed"}


def path_basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def path_extension(path: str) -> str:
    name = path_basename(path)
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def is_submission_artifact(path: str) -> bool:
    p = path.lower()
    name = path_basename(p)
    return (
        p.startswith("output/")
        and (name.endswith("_preds.json") or name.endswith("preds.json"))
    ) or (
        name.endswith("_preds.json")
        or name in {"all_preds.json", "preds.json", "prediction.json", "patch.txt"}
        or name.endswith(".patch")
        or name.endswith(".diff")
    )


def is_scratch_path(path: str) -> bool:
    p = path.lower()
    name = path_basename(p)
    scratch_names = {
        "test_issue.py",
        "comprehensive_test.py",
        "manual_test.py",
        "quick_test.py",
        "debug.py",
        "repro.py",
        "scratch.py",
    }
    if name in scratch_names:
        return True
    if re.match(r"^(debug|repro|scratch|tmp|temp)[_-].*", name):
        return True
    if re.match(r".*[_-](debug|repro|scratch|tmp|temp)\.[^.]+$", name):
        return True
    parts = p.split("/")
    return any(part in {"tmp", "temp", "scratch", ".scratch", "debug"} for part in parts)


def is_repo_test_path(path: str) -> bool:
    p = path.lower()
    name = path_basename(p)
    parts = p.split("/")
    if any(part in {"test", "tests", "testing", "spec", "specs"} for part in parts[:-1]):
        return True
    if name.startswith("test_") or name.endswith("_test.py") or name.endswith(".spec.js") or name.endswith(".test.js"):
        return True
    if name.endswith(".spec.ts") or name.endswith(".test.ts") or name.endswith(".spec.tsx") or name.endswith(".test.tsx"):
        return True
    return False


def classify_edit_path(path: str) -> str:
    if not path:
        return "other"
    if is_submission_artifact(path):
        return "submission_artifact"
    if is_scratch_path(path):
        return "scratch"
    if is_repo_test_path(path):
        return "test"
    if path_extension(path) in SOURCE_EXTENSIONS:
        return "source"
    return "other"


def edit_bucket_for_category(category: str) -> str:
    return {
        "source": "source_edits",
        "test": "test_edits",
        "scratch": "scratch_edits",
        "submission_artifact": "submission_artifacts",
        "other": "other_edits",
    }[category]


def extract_primary_path(kwargs: dict[str, Any], function: str) -> tuple[str, str]:
    if function == "Glob":
        return "", ""
    for key in ("file_path", "path", "notebook_path"):
        val = kwargs.get(key)
        if isinstance(val, str) and val.strip():
            return normalize_path_with_raw(val)
    return "", ""


class PathStore:
    def __init__(self) -> None:
        self._items: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    def add(self, path: str, raw_path: str, action: str, action_index: int, edit_category: str | None = None) -> None:
        if not path or is_workspace_root_path(path):
            return
        if path not in self._items:
            entry: dict[str, Any] = {
                "path": path,
                "raw_paths": [],
                "actions": [],
                "action_counts": OrderedDict(),
                "count": 0,
                "first_action_index": action_index,
            }
            if edit_category is not None:
                entry["edit_category"] = edit_category
            self._items[path] = entry
        entry = self._items[path]
        if raw_path and raw_path not in entry["raw_paths"]:
            entry["raw_paths"].append(raw_path)
        if action not in entry["action_counts"]:
            entry["actions"].append(action)
            entry["action_counts"][action] = 0
        entry["action_counts"][action] += 1
        entry["count"] += 1
        if action_index < entry["first_action_index"]:
            entry["first_action_index"] = action_index

    def values(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for entry in self._items.values():
            d = dict(entry)
            d["action_counts"] = dict(d["action_counts"])
            out.append(d)
        return out


# ---------------------------------------------------------------------------
# Command / failure classification
# ---------------------------------------------------------------------------


def command_references_submission_artifact(command: str) -> bool:
    c = normalize_text(command).replace("\\", "/").lower()
    return (
        "_preds.json" in c
        or "model_patch" in c
        or "prediction.json" in c
        or re.search(r"(?:^|[\s;&|])(?:\./)?(?:[^\s;&|]+/)*patch\.txt(?:[\s;&|]|$)", c) is not None
        or re.search(r"(?:^|[\s;&|])(?:\./)?(?:[^\s;&|]+/)*[^/\s;&|]+\.(?:patch|diff)(?:[\s;&|]|$)", c) is not None
        or re.search(r"(?:^|[\s;&|])(?:\./)?(?:[^\s;&|]+/)*output/?(?:[\s;&|]|$)", c) is not None
        or re.search(r"output/[^\s;]+preds\.json", c) is not None
    )


def command_is_environment_probe_noise(command: str) -> bool:
    c = normalize_text(command).strip()
    if "\n" in c:
        return False
    c = re.sub(r"\s+", " ", c).strip()
    c = re.sub(r"^(?:cd\s+\S+\s*&&\s*)+", "", c, flags=re.I).strip()
    if re.fullmatch(r"pwd", c, re.I):
        return True
    if re.fullmatch(r"echo\s+['\"]?testbed does not exist['\"]?", c, re.I):
        return True

    root = r"(?:\.|\./|testbed/?|\./testbed/?|(?:\./)?swebench_[^/\s]+/?(?:testbed/)?)"
    ls_root = rf"(?:pwd\s*&&\s*)?ls(?:\s+-[A-Za-z0-9]+)*(?:\s+{root})?"
    redir = r"(?:\s+2>/dev/null)?"
    pipe_suffix = r"(?:\s*\|\s*(?:head|grep)\b[^\n]*)?"
    fallback_echo = r"(?:\s*\|\|\s*echo\s+['\"]?testbed does not exist['\"]?)?"
    return re.fullmatch(ls_root + redir + pipe_suffix + fallback_echo, c, re.I) is not None


def command_is_git_state_noise(command: str) -> bool:
    c = normalize_text(command).strip()
    if "\n" in c:
        return False
    c = re.sub(r"\s+", " ", c).strip()
    c = re.sub(r"^(?:cd\s+\S+\s*&&\s*)+", "", c, flags=re.I).strip()
    git = r"git(?:\s+-C\s+\S+)?"
    patterns = (
        git + r"\s+status\b",
        git + r"\s+remote\b",
        git + r"\s+log\b",
        git + r"\s+branch\b",
        git + r"\s+show\b(?:\s+--stat\b|\s+head\b|\s+HEAD\b)",
    )
    return any(re.search(pattern, c, re.I) for pattern in patterns)


def text_is_missing_path_noise(text: str) -> bool:
    norm = normalize_text(text).strip()
    if not norm:
        return True
    lower = norm.lower()
    if lower == "no files found":
        return True
    if lower == "no testbed directory":
        return True
    if "<tool_use_error>path does not exist:" in lower:
        return True
    if "<tool_use_error>file does not exist." in lower:
        return True
    if "no such file or directory" in lower and any(
        token in lower for token in ("ls:", "find:", "grep:", "cat:", "sed:", "(eval):cd:")
    ):
        return True
    return False


def text_is_setup_failure_noise(text: str) -> bool:
    lower = normalize_text(text).lower()
    if "source checkout or from an editable installation without building the extension modules first" in lower:
        return True
    if "pip install -e ." in lower and "python setup.py build_ext --inplace" in lower:
        return True
    if "no module named 'extension_helpers'" in lower or 'no module named "extension_helpers"' in lower:
        return True
    if "setuptools==60.9.3 is used in combination with setuptools-scm>=8.x" in lower:
        return True
    return False


def text_is_environment_observation_noise(text: str) -> bool:
    return text_is_missing_path_noise(text) or text_is_setup_failure_noise(text)


def classify_command(command: str) -> str:
    cmd = normalize_text(command).strip()
    if command_references_submission_artifact(cmd):
        return "other"
    if command_is_git_state_noise(cmd) or command_is_environment_probe_noise(cmd):
        return "diagnostic"
    if _CMD_DESTRUCTIVE.search(cmd):
        return "destructive"
    if _CMD_TEST.search(cmd):
        return "test"
    if _CMD_BUILD_INSTALL.search(cmd):
        return "build_install"
    if _CMD_SETUP.search(cmd):
        return "setup"
    if _CMD_DIAGNOSTIC.search(cmd):
        return "diagnostic"
    return "other"


def command_bucket(command_class: str) -> str:
    return {
        "test": "test_commands",
        "diagnostic": "diagnostic_commands",
        "setup": "setup_commands",
        "destructive": "destructive_commands",
        "build_install": "build_install_commands",
        "other": "other_commands",
    }[command_class]


def extract_failure_lines(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in normalize_text(text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        for pattern, rule in _FAILURE_RULES:
            if pattern.match(line):
                rows.append({"line": line, "rule": rule})
                break
    return rows


def failure_bucket_for_action(action: dict[str, Any] | None, action_class: str | None) -> str:
    if action_class == "test":
        return "test_failures"
    if action_class == "build_install":
        return "install_failures"
    if action_class in {"setup", "destructive"}:
        return "setup_failures"
    if action_class in {"diagnostic", "other"}:
        return "runtime_failures"
    if action and action.get("class_") == "api_action":
        fn = action.get("function", "")
        if fn in {"Read", "Grep", "Glob"}:
            return "runtime_failures"
    return "other_failures"


def observation_kind_for_action(action: dict[str, Any], action_class: str | None) -> str | None:
    cls = action.get("class_")
    if cls == "api_action":
        fn = action.get("function", "")
        if fn == "Read":
            return "read_result"
        if fn == "Grep":
            return "grep_result"
        if fn == "Glob":
            return "glob_result"
    if cls == "code_action" and action.get("language") == "bash":
        if action_class == "test":
            return "test_output"
        if action_class == "diagnostic":
            return "diagnostic_output"
    return None


def find_preceding_action_index(content: list[dict[str, Any]], obs_idx: int) -> int:
    for i in range(obs_idx - 1, -1, -1):
        if content[i].get("class_") in {"api_action", "code_action"}:
            return i
    return -1


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def empty_commands() -> dict[str, list[dict[str, Any]]]:
    return {k: [] for k in COMMAND_BUCKETS}


def empty_failures() -> dict[str, list[dict[str, Any]]]:
    return {k: [] for k in _FAILURE_BUCKETS}


def empty_edits() -> dict[str, list[dict[str, Any]]]:
    return {k: [] for k in EDIT_BUCKETS}


def make_command_entry(idx: int, item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw = strip_harness_prompt_junk(item.get("content") or "")
    excerpt, truncated = truncate_text(raw, args.max_command_chars)
    cls = classify_command(raw)
    entry = {
        "action_index": idx,
        "command_class": cls,
        "command": excerpt,
        "sha256": sha256_text(raw),
        "truncated": truncated,
        "contains_heredoc": bool(_HEREDOC.search(raw)),
        "prior_agent_description": short_description(item.get("description") or ""),
    }
    add_original_index(entry, "original_action_index", item, idx)
    return entry


def make_observation_evidence(
    *,
    action_index: int,
    observation_index: int,
    kind: str,
    observation_text: str,
    action: dict[str, Any],
    observation_item: dict[str, Any],
    action_class: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    norm = strip_harness_prompt_junk(observation_text)
    excerpt, truncated = truncate_text(norm, args.max_observation_excerpt_chars)
    tool_or_class = ""
    evidence_path = ""
    excluded_from_rendered_text = False
    if action.get("class_") == "api_action":
        tool_or_class = str(action.get("function") or "")
        kwargs = action.get("kwargs") if isinstance(action.get("kwargs"), dict) else {}
        evidence_path, _ = extract_primary_path(kwargs, tool_or_class)
        excluded_from_rendered_text = is_submission_artifact(evidence_path)
    elif action.get("class_") == "code_action":
        tool_or_class = action_class or "bash"
    excluded_from_rendered_text = excluded_from_rendered_text or text_is_environment_observation_noise(norm)
    entry = {
        "source_action_index": action_index,
        "observation_index": observation_index,
        "kind": kind,
        "tool_or_command_class": tool_or_class,
        "path": evidence_path,
        "excluded_from_rendered_text": excluded_from_rendered_text,
        "excerpt": excerpt,
        "sha256": sha256_text(norm),
        "truncated": truncated,
    }
    add_original_index(entry, "original_action_index", action, action_index)
    add_original_index(entry, "original_observation_index", observation_item, observation_index)
    return entry


def make_edit_entries(idx: int, item: dict[str, Any], fn: str, kwargs: dict[str, Any], path: str, raw_path: str, description: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    category = classify_edit_path(path)
    base = {
        "action_index": idx,
        "tool": fn,
        "path": path,
        "raw_path": raw_path,
        "edit_category": category,
        "prior_agent_description": description,
    }
    add_original_index(base, "original_action_index", item, idx)
    entries: list[dict[str, Any]] = []

    def one(old_value: Any, new_value: Any, subedit_index: int | None = None) -> dict[str, Any]:
        old_raw = strip_harness_prompt_junk(old_value)
        new_raw = strip_harness_prompt_junk(new_value)
        old_excerpt, old_truncated = truncate_text(old_raw, args.max_edit_excerpt_chars)
        new_excerpt, new_truncated = truncate_text(new_raw, args.max_edit_excerpt_chars)
        if args.edit_body_mode == "hashes-only":
            old_excerpt = ""
            new_excerpt = ""
            old_truncated = False
            new_truncated = False
        entry = dict(base)
        if subedit_index is not None:
            entry["subedit_index"] = subedit_index
        entry.update(
            {
                "old_text_sha256": sha256_text(old_raw) if old_raw else "",
                "new_text_sha256": sha256_text(new_raw) if new_raw else "",
                "old_text_excerpt": old_excerpt,
                "new_text_excerpt": new_excerpt,
                "truncated": old_truncated or new_truncated,
            }
        )
        return entry

    if fn == "Edit":
        entries.append(one(kwargs.get("old_string") or "", kwargs.get("new_string") or ""))
    elif fn == "Write":
        entries.append(one("", kwargs.get("content") or ""))
    elif fn == "MultiEdit":
        sub_edits = kwargs.get("edits") or []
        if isinstance(sub_edits, list):
            for sub_idx, sub in enumerate(sub_edits):
                if isinstance(sub, dict):
                    entries.append(one(sub.get("old_string") or "", sub.get("new_string") or "", sub_idx))
    return entries


def first_user_prompt(content: Iterable[dict[str, Any]]) -> str:
    for item in content:
        if item.get("class_") == "text_observation" and item.get("source") == "user":
            text = strip_harness_prompt_junk(item.get("content") or "")
            if text.strip():
                return text
    return ""


def final_message(content: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    for item in reversed(content):
        if item.get("class_") == "message_action":
            raw = strip_harness_prompt_junk(item.get("content") or "")
            excerpt, truncated = truncate_text(raw, args.max_final_message_chars)
            return {
                "text": excerpt,
                "sha256": sha256_text(raw),
                "truncated": truncated,
                "included_in_rendered_text": bool(excerpt),
            }
    return {"text": "", "sha256": "", "truncated": False, "included_in_rendered_text": False}


def renderer_flags(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "template",
        "include_todowrite",
        "include_reasoning",
        "max_problem_chars",
        "max_raw_user_prompt_chars",
        "max_final_message_chars",
        "max_edit_excerpt_chars",
        "edit_body_mode",
        "max_command_chars",
        "max_observed_failure_lines",
        "max_observation_excerpt_chars",
        "max_observation_evidence_items",
        "max_rendered_commands",
        "max_rendered_command_total_chars",
        "max_rendered_command_chars",
        "exclude_setup_commands",
        "exclude_destructive_commands",
        "exclude_build_install_commands",
        "keep_excluded_edit_details",
    ]
    return {k: getattr(args, k) for k in keys}


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------


def convert_trajectory(traj: dict[str, Any], args: argparse.Namespace, *, record_index: int, input_label: str = "") -> dict[str, Any]:
    details = traj.get("details") if isinstance(traj.get("details"), dict) else {}
    issue_meta = details.get("issue_metadata") if isinstance(details.get("issue_metadata"), dict) else {}
    models = details.get("models") if isinstance(details.get("models"), list) else []
    content = traj.get("content") if isinstance(traj.get("content"), list) else []

    warnings: list[str] = []
    for w in details.get("warnings") or []:
        warnings.append(f"[source] {normalize_text(w)}")

    source_metadata = {
        "repo": normalize_text(issue_meta.get("repo") or ""),
        "instance_id": normalize_text(issue_meta.get("instance_id") or ""),
        "base_commit": normalize_text(issue_meta.get("base_commit") or ""),
        "model": normalize_text(models[0]) if models else "",
        "source_file": normalize_text(details.get("source_file") or input_label),
    }

    raw_prompt = first_user_prompt(content)
    problem_statement_raw = strip_harness_prompt_junk(issue_meta.get("problem_statement") or "")
    if problem_statement_raw:
        problem_source = "issue_metadata.problem_statement"
    elif raw_prompt:
        problem_statement_raw = raw_prompt
        problem_source = "first_user_text_observation"
        warnings.append("prior_problem.problem_statement fell back to first user text_observation")
    else:
        problem_source = "missing"
        warnings.append("prior_problem.problem_statement is empty")

    problem_text, problem_truncated = truncate_text(problem_statement_raw, args.max_problem_chars)
    raw_prompt_text, raw_prompt_truncated = truncate_text(raw_prompt, args.max_raw_user_prompt_chars)
    prior_problem = {
        "problem_statement": problem_text,
        "raw_user_prompt": raw_prompt_text,
        "source": problem_source,
        "sha256": sha256_text(problem_statement_raw),
        "raw_user_prompt_sha256": sha256_text(raw_prompt) if raw_prompt else "",
        "truncated": problem_truncated,
        "raw_user_prompt_truncated": raw_prompt_truncated,
    }

    has_reasoning = any("reasoning_content" in item for item in content if isinstance(item, dict))
    has_todowrite = any(
        item.get("class_") == "api_action" and item.get("function") == "TodoWrite"
        for item in content
        if isinstance(item, dict)
    )

    inspected = PathStore()
    edited_files = PathStore()
    search_anchors: list[dict[str, Any]] = []
    commands = empty_commands()
    observed_failures = empty_failures()
    edits = empty_edits()
    observation_evidence: list[dict[str, Any]] = []
    planning_actions: list[dict[str, Any]] = []
    reasoning: list[dict[str, Any]] = []

    action_by_index: dict[int, dict[str, Any]] = {}
    action_class_by_index: dict[int, str] = {}
    failure_seen: set[tuple[str, str]] = set()
    failure_count = 0
    unknown_type_count = 0
    excluded_submission_artifacts_count = 0
    excluded_scratch_edits_count = 0
    observation_evidence_skipped_due_to_cap = 0

    for idx, item in enumerate(content):
        if not isinstance(item, dict):
            unknown_type_count += 1
            continue

        cls = item.get("class_", "")

        if "reasoning_content" in item:
            reason_raw = strip_harness_prompt_junk(item.get("reasoning_content") or "")
            if args.include_reasoning and reason_raw:
                reason_text, reason_truncated = truncate_text(reason_raw, args.max_reasoning_chars)
                entry = {
                    "action_index": idx,
                    "item_class": cls,
                    "text": reason_text,
                    "sha256": sha256_text(reason_raw),
                    "truncated": reason_truncated,
                }
                add_original_index(entry, "original_action_index", item, idx)
                reasoning.append(entry)

        if cls in {"api_action", "code_action"}:
            action_by_index[idx] = item

        if cls == "api_action":
            fn = str(item.get("function") or "")
            kwargs = item.get("kwargs") if isinstance(item.get("kwargs"), dict) else {}
            description = short_description(item.get("description") or "")

            if fn in PLANNING_TOOLS:
                if args.include_todowrite:
                    payload_raw = json.dumps(canonicalize_json_value(kwargs), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    payload_excerpt, payload_truncated = truncate_text(payload_raw, args.max_planning_action_chars)
                    entry = {
                        "action_index": idx,
                        "tool": fn,
                        "payload": payload_excerpt,
                        "sha256": sha256_text(payload_raw),
                        "truncated": payload_truncated,
                        "prior_agent_description": description,
                    }
                    add_original_index(entry, "original_action_index", item, idx)
                    planning_actions.append(entry)
                continue

            if fn in INSPECT_TOOLS:
                path, raw_path = extract_primary_path(kwargs, fn)
                if path:
                    inspected.add(path, raw_path, fn, idx)
                elif fn == "Read":
                    warnings.append(f"[{idx}] Read action has no extractable file path")

                if fn in SEARCH_ANCHOR_TOOLS:
                    raw_anchor_path = kwargs.get("path") or ""
                    norm_anchor_path = normalize_path(raw_anchor_path) if isinstance(raw_anchor_path, str) else ""
                    pattern = normalize_text(kwargs.get("pattern") or "")
                    glob = normalize_text(kwargs.get("glob") or "")
                    if fn == "Glob":
                        glob = pattern
                        pattern = ""
                    entry = {
                        "action_index": idx,
                        "tool": fn,
                        "pattern": pattern,
                        "path": norm_anchor_path,
                        "raw_path": normalize_text(raw_anchor_path),
                        "glob": glob,
                        "prior_agent_description": description,
                    }
                    add_original_index(entry, "original_action_index", item, idx)
                    search_anchors.append(entry)

            elif fn in EDIT_TOOLS:
                path, raw_path = extract_primary_path(kwargs, fn)
                category = classify_edit_path(path)
                if path:
                    edited_files.add(path, raw_path, fn, idx, edit_category=category)
                else:
                    warnings.append(f"[{idx}] {fn} action has no extractable file path")
                entries = make_edit_entries(idx, item, fn, kwargs, path, raw_path, description, args)
                if fn == "MultiEdit" and not entries:
                    warnings.append(f"[{idx}] MultiEdit has no extractable edits")
                for entry in entries:
                    bucket = edit_bucket_for_category(entry["edit_category"])
                    if bucket == "submission_artifacts":
                        excluded_submission_artifacts_count += 1
                    if bucket == "scratch_edits":
                        excluded_scratch_edits_count += 1
                    if args.keep_excluded_edit_details or bucket in {"source_edits", "test_edits", "other_edits"}:
                        edits[bucket].append(entry)
            else:
                if fn:
                    unknown_type_count += 1
                else:
                    warnings.append(f"[{idx}] api_action has empty function")

        elif cls == "code_action":
            if item.get("language") == "bash":
                entry = make_command_entry(idx, item, args)
                ccls = entry["command_class"]
                action_class_by_index[idx] = ccls
                commands[command_bucket(ccls)].append(entry)

        elif cls == "text_observation":
            if item.get("source") == "environment":
                obs_text = strip_harness_prompt_junk(item.get("content") or "")
                prev_idx = find_preceding_action_index(content, idx)
                prev_action = action_by_index.get(prev_idx)
                prev_class = action_class_by_index.get(prev_idx)

                if prev_action is not None:
                    kind = observation_kind_for_action(prev_action, prev_class)
                    if kind and obs_text and args.max_observation_evidence_items > 0:
                        if len(observation_evidence) < args.max_observation_evidence_items:
                            observation_evidence.append(
                                make_observation_evidence(
                                    action_index=prev_idx,
                                    observation_index=idx,
                                    kind=kind,
                                    observation_text=obs_text,
                                    action=prev_action,
                                    observation_item=item,
                                    action_class=prev_class,
                                    args=args,
                                )
                            )
                        else:
                            observation_evidence_skipped_due_to_cap += 1

                if args.max_observed_failure_lines > 0:
                    bucket = failure_bucket_for_action(prev_action, prev_class)
                    for failure in extract_failure_lines(obs_text):
                        key = (bucket, failure["line"])
                        if key in failure_seen:
                            continue
                        failure_seen.add(key)
                        if failure_count >= args.max_observed_failure_lines:
                            break
                        failure_entry = {
                            "observation_index": idx,
                            "source_action_index": prev_idx,
                            "line": failure["line"],
                            "rule": failure["rule"],
                        }
                        if prev_action is not None:
                            add_original_index(failure_entry, "original_action_index", prev_action, prev_idx)
                        add_original_index(failure_entry, "original_observation_index", item, idx)
                        observed_failures[bucket].append(failure_entry)
                        failure_count += 1
            # User observations handled by prior_problem only.

        elif cls == "message_action":
            pass
        else:
            if cls:
                unknown_type_count += 1

    if failure_count >= args.max_observed_failure_lines and args.max_observed_failure_lines > 0:
        warnings.append(f"observed_failures capped at max_observed_failure_lines={args.max_observed_failure_lines}")
    if observation_evidence_skipped_due_to_cap > 0:
        warnings.append(
            f"observation_evidence capped at max_observation_evidence_items={args.max_observation_evidence_items}; "
            f"skipped={observation_evidence_skipped_due_to_cap}"
        )
    if unknown_type_count:
        warnings.append(f"Skipped {unknown_type_count} unhandled item classes/functions")

    prior_agent_final_message = final_message(content, args)

    renderer = {
        "name": SCRIPT_NAME,
        "memory_schema_version": MEMORY_SCHEMA_VERSION,
        "template": args.template,
        "flags": renderer_flags(args),
        "source_trajectory_sha256": sha256_json(traj),
        "input_content_sha256": sha256_json(content),
    }

    quality = {
        "warnings": warnings,
        "excluded_reasoning_content": has_reasoning and not args.include_reasoning,
        "excluded_todowrite": has_todowrite and not args.include_todowrite,
        "excluded_submission_artifacts_count": excluded_submission_artifacts_count,
        "excluded_scratch_edits_count": excluded_scratch_edits_count,
        "unknown_type_count": unknown_type_count,
    }

    memory: dict[str, Any] = {
        "memory_schema_version": MEMORY_SCHEMA_VERSION,
        "source_trajectory_id": normalize_text(traj.get("id") or ""),
        "source_schema_version": normalize_text(traj.get("schema_version") or ""),
        "renderer": renderer,
        "source_metadata": source_metadata,
        "prior_problem": prior_problem,
        "files": {
            "inspected": inspected.values(),
            "edited": edited_files.values(),
        },
        "search_anchors": search_anchors,
        "commands": commands,
        "observed_failures": observed_failures,
        "observation_evidence": observation_evidence,
        "edits": edits,
        "planning_actions": planning_actions,
        "reasoning": reasoning,
        "prior_agent_final_message": prior_agent_final_message,
        "rendered_text": "",
        "quality": quality,
    }
    memory["rendered_text"] = render_memory(memory, args)
    return memory


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def section(title: str) -> list[str]:
    return ["", title, "-" * len(title)]


def indent_block(text: str, prefix: str = "  ") -> list[str]:
    if text == "":
        return [prefix.rstrip()]
    return [prefix + line for line in text.split("\n")]


def counts_label(action_counts: dict[str, int]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in action_counts.items())


def command_render_text(command: str, max_chars: int) -> tuple[str, bool]:
    return truncate_text(strip_harness_prompt_junk(command), max_chars)


def rendered_commands(memory: dict[str, Any], args: argparse.Namespace) -> list[str]:
    selected: list[tuple[str, dict[str, Any]]] = []
    for bucket, label in (
        ("test_commands", "test"),
        ("diagnostic_commands", "diagnostic"),
        ("other_commands", "other/reproduction"),
        ("build_install_commands", "build/install"),
        ("setup_commands", "setup"),
        ("destructive_commands", "destructive"),
    ):
        if bucket == "setup_commands" and args.exclude_setup_commands:
            continue
        if bucket == "destructive_commands" and args.exclude_destructive_commands:
            continue
        if bucket == "build_install_commands" and args.exclude_build_install_commands:
            continue
        for entry in memory["commands"][bucket]:
            selected.append((label, entry))
    selected.sort(key=lambda pair: pair[1]["action_index"])

    lines: list[str] = []
    total_chars = 0
    count = 0
    for label, entry in selected:
        if count >= args.max_rendered_commands:
            break
        if command_is_git_state_noise(entry["command"]) or command_is_environment_probe_noise(entry["command"]):
            continue
        cmd, truncated = command_render_text(entry["command"], args.max_rendered_command_chars)
        line = f"- [{label}] {cmd}"
        if total_chars + len(line) > args.max_rendered_command_total_chars:
            remaining = max(args.max_rendered_command_total_chars - total_chars, 0)
            if remaining <= 0:
                break
            line = line[:remaining]
            truncated = True
        lines.append(line)
        if truncated or entry.get("truncated"):
            lines.append("  [... truncated ...]")
        total_chars += len(line)
        count += 1
    return lines


def render_memory(memory: dict[str, Any], args: argparse.Namespace) -> str:
    lines: list[str] = []
    lines.append("PRIOR EXPERIENCE MEMORY")
    lines.append(_TEMPLATE_HEADERS.get(args.template, _TEMPLATE_HEADERS["generic"]))

    lines.extend(section("SOURCE"))
    src = memory["source_metadata"]
    lines.append(f"- trajectory_id: {memory['source_trajectory_id']}")
    lines.append(f"- repo: {src.get('repo', '')}")
    lines.append(f"- instance_id: {src.get('instance_id', '')}")
    lines.append(f"- base_commit: {src.get('base_commit', '')}")
    lines.append(f"- model: {src.get('model', '')}")

    lines.extend(section("PRIOR PROBLEM"))
    problem = memory["prior_problem"]
    if problem.get("problem_statement"):
        lines.append(strip_harness_prompt_junk(problem["problem_statement"]))
        if problem.get("truncated"):
            lines.append("[... truncated ...]")
    else:
        lines.append("(not found)")

    lines.extend(section("FILES INSPECTED"))
    inspected = memory["files"]["inspected"]
    rendered_inspected = [item for item in inspected if not is_submission_artifact(item.get("path", ""))]
    if rendered_inspected:
        for item in rendered_inspected:
            label = counts_label(item.get("action_counts", {}))
            lines.append(f"- {strip_harness_prompt_junk(item['path'])} [{label}]")
    else:
        lines.append("(none)")

    lines.extend(section("SEARCH ANCHORS"))
    anchors = memory["search_anchors"]
    if anchors:
        for anchor in anchors:
            parts = [f"- [{anchor['action_index']}] {anchor['tool']}"]
            if anchor.get("pattern"):
                parts.append(f"pattern={strip_harness_prompt_junk(anchor['pattern'])!r}")
            if anchor.get("path"):
                parts.append(f"path={strip_harness_prompt_junk(anchor['path'])!r}")
            if anchor.get("glob"):
                parts.append(f"glob={strip_harness_prompt_junk(anchor['glob'])!r}")
            desc = anchor.get("prior_agent_description") or ""
            if desc and len(desc) <= 160:
                parts.append(f"prior_agent_description={strip_harness_prompt_junk(desc)!r}")
            lines.append(" ".join(parts))
    else:
        lines.append("(none)")

    lines.extend(section("TEST / DIAGNOSTIC / OTHER COMMANDS"))
    command_lines = rendered_commands(memory, args)
    lines.extend(command_lines if command_lines else ["(none)"])

    lines.extend(section("OBSERVED TEST / RUNTIME FAILURES"))
    failure_lines: list[str] = []
    for bucket in ("test_failures", "runtime_failures"):
        for failure in memory["observed_failures"].get(bucket, []):
            failure_lines.append(f"- {strip_harness_prompt_junk(failure['line'])}")
    lines.extend(failure_lines if failure_lines else ["(none)"])

    lines.extend(section("RELEVANT OBSERVATION EXCERPTS"))
    evidence = [ev for ev in memory["observation_evidence"] if not ev.get("excluded_from_rendered_text")]
    if evidence:
        for ev in evidence:
            heading = ev["kind"].replace("_", " ").title()
            lines.append(f"- [{heading} after action #{ev['source_action_index']}]")
            lines.extend(indent_block(strip_harness_prompt_junk(ev["excerpt"]), "  "))
            if ev.get("truncated"):
                lines.append("  [... truncated ...]")
    else:
        lines.append("(none)")

    lines.extend(section("SOURCE EDIT EVIDENCE"))
    visible_edits = list(memory["edits"].get("source_edits", [])) + list(memory["edits"].get("test_edits", []))
    if visible_edits:
        for edit in visible_edits:
            lines.append(f"- {strip_harness_prompt_junk(edit['path'])}")
            lines.append(f"  Tool: {edit['tool']} action #{edit['action_index']}")
            if edit.get("prior_agent_description") and len(edit["prior_agent_description"]) <= 160:
                lines.append(f"  prior_agent_description: {strip_harness_prompt_junk(edit['prior_agent_description'])}")
            if edit.get("old_text_sha256"):
                lines.append(f"  old_text_sha256: {edit['old_text_sha256']}")
            if edit.get("new_text_sha256"):
                lines.append(f"  new_text_sha256: {edit['new_text_sha256']}")
            if edit.get("old_text_excerpt"):
                lines.append("  old_text_excerpt:")
                lines.extend(indent_block(strip_harness_prompt_junk(edit["old_text_excerpt"]), "    "))
            if edit.get("new_text_excerpt"):
                lines.append("  new_text_excerpt:")
                lines.extend(indent_block(strip_harness_prompt_junk(edit["new_text_excerpt"]), "    "))
            if edit.get("truncated"):
                lines.append("    [... truncated ...]")
    else:
        lines.append("(none)")

    lines.extend(section("PRIOR AGENT FINAL MESSAGE"))
    fm = memory["prior_agent_final_message"]
    if fm.get("text"):
        lines.append("The following is prior-agent generated text, not ground truth:")
        lines.append(strip_harness_prompt_junk(fm["text"]))
        if fm.get("truncated"):
            lines.append("[... truncated ...]")
    else:
        lines.append("(none)")

    lines.extend(section("INJECTION WARNING"))
    lines.append(INJECTION_WARNING)
    return strip_harness_prompt_junk("\n".join(lines))


# ---------------------------------------------------------------------------
# I/O and CLI
# ---------------------------------------------------------------------------


def safe_filename_component(value: str) -> str:
    value = normalize_text(value).strip() or "unknown"
    value = re.sub(r"[/\\:*?\"<>|\s]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    return value[:120] or "unknown"


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: JSON parse error: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object per line, got {type(obj).__name__}")
            yield lineno, obj


def validate_limits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    limit_names = [
        "max_problem_chars",
        "max_raw_user_prompt_chars",
        "max_final_message_chars",
        "max_edit_excerpt_chars",
        "max_command_chars",
        "max_observed_failure_lines",
        "max_observation_excerpt_chars",
        "max_observation_evidence_items",
        "max_rendered_commands",
        "max_rendered_command_total_chars",
        "max_rendered_command_chars",
        "max_planning_action_chars",
        "max_reasoning_chars",
    ]
    for name in limit_names:
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ADP trajectory JSONL to deterministic prior experience memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              %(prog)s claude_code_adp.jsonl
              %(prog)s claude_code_adp.jsonl --output-jsonl out/memories.jsonl --output-text-dir out/text
              %(prog)s claude_code_adp.jsonl --template openhands --stats
            """
        ),
    )
    parser.add_argument("input", type=Path, help="ADP trajectory JSONL file.")
    parser.add_argument("--output-jsonl", type=Path, default=Path("memories.jsonl"), metavar="FILE")
    parser.add_argument("--output-text-dir", type=Path, default=Path("memories"), metavar="DIR")
    parser.add_argument("--template", choices=["generic", "swe-agent", "openhands"], default="generic")

    parser.add_argument("--include-todowrite", action="store_true", help="Include TodoWrite payloads in planning_actions JSON section. Not rendered by default.")
    parser.add_argument("--include-reasoning", action="store_true", help="Include reasoning_content in reasoning JSON section for ablations. Not rendered by default.")
    parser.add_argument("--keep-excluded-edit-details", action=argparse.BooleanOptionalAction, default=True, help="Keep scratch/submission edit details in JSON for auditability (default: true).")

    parser.add_argument("--exclude-setup-commands", action=argparse.BooleanOptionalAction, default=True, help="Suppress setup commands from rendered text (default: true).")
    parser.add_argument("--exclude-destructive-commands", action=argparse.BooleanOptionalAction, default=True, help="Suppress destructive commands from rendered text (default: true).")
    parser.add_argument("--exclude-build-install-commands", action=argparse.BooleanOptionalAction, default=True, help="Suppress build/install boilerplate commands from rendered text (default: true).")

    parser.add_argument("--max-problem-chars", type=int, default=4000)
    parser.add_argument("--max-raw-user-prompt-chars", type=int, default=8000)
    parser.add_argument("--max-final-message-chars", type=int, default=2000)
    parser.add_argument("--max-edit-excerpt-chars", type=int, default=1200)
    parser.add_argument("--edit-body-mode", choices=["hashes-only", "excerpts"], default="hashes-only", help="Whether edit evidence includes old/new excerpts or only hashes (default: hashes-only).")
    parser.add_argument("--max-command-chars", type=int, default=2000)
    parser.add_argument("--max-observed-failure-lines", type=int, default=80)
    parser.add_argument("--max-observation-excerpt-chars", type=int, default=1200)
    parser.add_argument("--max-observation-evidence-items", type=int, default=20)
    parser.add_argument("--max-rendered-commands", type=int, default=12)
    parser.add_argument("--max-rendered-command-total-chars", type=int, default=4000)
    parser.add_argument("--max-rendered-command-chars", type=int, default=500)
    parser.add_argument("--max-planning-action-chars", type=int, default=2000)
    parser.add_argument("--max-reasoning-chars", type=int, default=2000)
    parser.add_argument("--stats", action="store_true", help="Print per-record stats to stderr.")

    args = parser.parse_args(argv)
    validate_limits(args, parser)
    return args


def convert_file(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_text_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    try:
        with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as out_jsonl:
            for count, (lineno, traj) in enumerate(iter_jsonl(args.input), start=1):
                try:
                    memory = convert_trajectory(traj, args, record_index=count, input_label=args.input.name)
                except Exception as exc:  # noqa: BLE001 - emit deterministic error record for auditability
                    traj_id = normalize_text(traj.get("id") or f"line-{lineno}")
                    memory = {
                        "memory_schema_version": MEMORY_SCHEMA_VERSION,
                        "source_trajectory_id": traj_id,
                        "source_schema_version": normalize_text(traj.get("schema_version") or ""),
                        "renderer": {
                            "name": SCRIPT_NAME,
                            "memory_schema_version": MEMORY_SCHEMA_VERSION,
                            "template": args.template,
                            "flags": renderer_flags(args),
                            "source_trajectory_sha256": sha256_json(traj),
                            "input_content_sha256": sha256_json(traj.get("content") or []),
                        },
                        "source_metadata": {},
                        "prior_problem": {"problem_statement": "", "raw_user_prompt": "", "sha256": "", "truncated": False},
                        "files": {"inspected": [], "edited": []},
                        "search_anchors": [],
                        "commands": empty_commands(),
                        "observed_failures": empty_failures(),
                        "observation_evidence": [],
                        "edits": empty_edits(),
                        "planning_actions": [],
                        "reasoning": [],
                        "prior_agent_final_message": {"text": "", "sha256": "", "truncated": False, "included_in_rendered_text": False},
                        "rendered_text": "",
                        "quality": {
                            "warnings": [f"CONVERTER ERROR: {type(exc).__name__}: {exc}"],
                            "excluded_reasoning_content": False,
                            "excluded_todowrite": False,
                            "excluded_submission_artifacts_count": 0,
                            "excluded_scratch_edits_count": 0,
                            "unknown_type_count": 0,
                        },
                    }
                out_jsonl.write(stable_json_dumps(memory) + "\n")
                safe_id = safe_filename_component(memory.get("source_trajectory_id") or "unknown")
                text_path = args.output_text_dir / f"{count:06d}_{safe_id}.txt"
                text_path.write_text(memory.get("rendered_text", "") + "\n", encoding="utf-8", newline="\n")
                if args.stats:
                    print(
                        f"{count:06d} {memory.get('source_trajectory_id','')}: "
                        f"source_edits={len(memory['edits']['source_edits'])} "
                        f"test_edits={len(memory['edits']['test_edits'])} "
                        f"scratch_edits={len(memory['edits']['scratch_edits'])} "
                        f"artifacts={len(memory['edits']['submission_artifacts'])} "
                        f"warnings={len(memory['quality']['warnings'])}",
                        file=sys.stderr,
                    )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if count == 0:
        print("WARNING: input JSONL is empty; wrote empty output JSONL", file=sys.stderr)
    print(f"Wrote {count} memory records to {args.output_jsonl}", file=sys.stderr)
    print(f"Wrote {count} text files to {args.output_text_dir}/", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return convert_file(args)


if __name__ == "__main__":
    raise SystemExit(main())
