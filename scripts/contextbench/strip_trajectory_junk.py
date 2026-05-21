#!/usr/bin/env python3
"""
Deterministically strip repeated environment / harness junk from SWE-agent
trajectory representations.

This script is designed for the experiment:

    Raw Claude Code JSONL
        -> converted ADP JSONL
        -> deterministic prior-experience memory

The important design choice is that the ADP trajectory is treated as the
canonical event sequence. A deterministic strip manifest is built from ADP
content indices. The stripped ADP and stripped memory are then derived from the
same keep/drop decisions.

Raw Claude JSONL can also be rendered into a stripped transcript using the same
path/command/text policy. Exact raw<->ADP event masking requires provenance from
Raw->ADP conversion, so this script records a warning when provenance is absent.

No LLM calls, embeddings, network access, random IDs, wall-clock timestamps, or
filesystem-dependent path resolution are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Iterator

STRIP_POLICY_VERSION = "1.2.0"
SCRIPT_NAME = "strip_trajectory_junk.py"

# ---------------------------------------------------------------------------
# Stable JSON / hashing
# ---------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
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


def is_edit_tool_result_observation(content: list[dict[str, Any]], idx: int) -> bool:
    item = content[idx]
    if item.get("class_") != "text_observation" or item.get("source") != "environment":
        return False
    prev = nearest_preceding_action([], content, idx)
    if prev < 0:
        return False
    prev_item = content[prev]
    if prev_item.get("class_") != "api_action":
        return False
    return normalize_text(prev_item.get("function") or "") in EDIT_TOOLS


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


def strip_system_reminders_from_value(value: Any) -> Any:
    if isinstance(value, str):
        return strip_system_reminders(value)
    if isinstance(value, list):
        return [strip_system_reminders_from_value(v) for v in value]
    if isinstance(value, tuple):
        return [strip_system_reminders_from_value(v) for v in value]
    if isinstance(value, dict):
        return {k: strip_system_reminders_from_value(v) for k, v in value.items()}
    return value


def text_contains_hard_forbidden_payload(text: Any) -> bool:
    return bool(HARD_FORBIDDEN_TOKENS_RE.search(normalize_text(text)))


def command_contains_patch_payload(command: Any) -> bool:
    cmd = normalize_text(command)
    if "diff --git" in cmd.lower() or "```diff" in cmd.lower():
        return True
    if text_contains_hard_forbidden_payload(cmd):
        return True
    match = PATCH_WRITE_RE.search(cmd)
    if not match:
        return False
    target = next((g for g in match.groups() if g), "")
    target = target.strip().strip("'\"")
    base = target.rsplit("/", 1)[-1].lower()
    return (
        base in {"patch", "diff"}
        or base == "patch.txt"
        or base.endswith(".patch")
        or base.endswith(".diff")
        or base.startswith("patch.")
        or base.startswith("diff.")
    )


def canonicalize(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [canonicalize(v) for v in value]
    if isinstance(value, tuple):
        return [canonicalize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): canonicalize(value[k]) for k in sorted(value.keys(), key=str)}
    return value


def stable_json_dumps(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(canonicalize(value), ensure_ascii=False, sort_keys=True, indent=2)
    return json.dumps(canonicalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(stable_json_dumps(value))


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars < 0:
        raise ValueError("max_chars must be non-negative")
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


# ---------------------------------------------------------------------------
# Path / command classifiers
# ---------------------------------------------------------------------------

SOURCE_EXTENSIONS = frozenset(
    {
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs",
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cs", ".rb", ".php",
        ".swift", ".kt", ".kts", ".scala", ".sh", ".bash", ".zsh", ".sql",
        ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json", ".xml", ".html",
        ".css", ".scss", ".md", ".rst",
    }
)

INSPECT_TOOLS = frozenset({"Read", "Grep", "Glob"})
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit"})
PLANNING_TOOLS = frozenset({"TodoWrite"})

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
    r"\./gradlew\s+test|rspec|bundle\s+exec\s+rspec|python3?\s+[^\n;]*test[^\n;]*\.py|python\s+[^\n;]*test[^\n;]*\.py)\b"
)
_CMD_BUILD_INSTALL = re.compile(
    r"(?is)\b(pip3?\s+install|python3?\s+-m\s+pip\s+install|uv\s+pip\s+install|"
    r"poetry\s+install|pipenv\s+install|npm\s+install|npm\s+ci|pnpm\s+install|"
    r"yarn\s+install|cargo\s+build|cargo\s+check|npm\s+run\s+build|pnpm\s+build|"
    r"yarn\s+build|mvn\s+(install|package|compile)|gradle\s+build|\./gradlew\s+build|"
    r"make(\s|$)|python3?\s+setup\.py\s+(build|install|develop|build_ext))\b"
)
_CMD_DIAGNOSTIC = re.compile(
    r"(?is)^\s*(cd\s+\S+\s*&&\s*)?(ls|find|grep|rg|sed|cat|head|tail|wc|pwd|"
    r"git\s+status|git\s+diff|git\s+log|git\s+show|git\s+grep|python\s+-c)\b"
)

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

RAW_TRANSPORT_KEYS = frozenset(
    {
        "uuid", "parentUuid", "sessionId", "version", "gitBranch", "cwd", "userType",
        "requestId", "messageId", "isSidechain", "isCompactSummary", "isMeta",
        "isApiErrorMessage", "isSnapshotUpdate", "timestamp",
    }
)
ADP_DETAILS_DROP_KEYS = frozenset(
    {
        "raw_tool_inputs_by_id", "raw_tool_results_by_id", "file_history_snapshots",
        "cwd", "git_branches", "tool_use_id_to_function", "summary",
    }
)


def normalize_path_with_raw(path_value: Any) -> tuple[str, str]:
    raw = normalize_text(path_value).strip()
    p = raw.replace("\\", "/").strip()
    p = re.sub(r"/+", "/", p)
    while p.startswith("./"):
        p = p[2:]

    # /tmp/.../testbed/foo.py -> foo.py
    m_abs = re.search(r"(?:^|/)(?:testbed)/(.*)$", p)
    if m_abs and (p.startswith("/") or "/tmp/" in p or p.startswith("tmp/")):
        p = m_abs.group(1)

    p = re.sub(r"^(?:[^/]*swebench[^/]*/)+testbed/", "", p)
    p = re.sub(r"^swebench_[^/]+/testbed/", "", p)
    if p.startswith("testbed/"):
        p = p[len("testbed/") :]
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
    return normalize_text(path).replace("\\", "/").rsplit("/", 1)[-1]


def path_extension(path: str) -> str:
    name = path_basename(path)
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def is_submission_artifact_path(path: str) -> bool:
    p = normalize_path(path).lower()
    name = path_basename(p)
    return (
        p.startswith("output/")
        or name.endswith("_preds.json")
        or name in {"all_preds.json", "preds.json", "prediction.json"}
        or name == "patch.txt"
        or name.endswith(".patch")
        or name.endswith(".diff")
    )


def is_harness_path(path: str) -> bool:
    p = normalize_path(path).lower()
    name = path_basename(p)
    if name in {"manual.yaml", "manual.yml"}:
        return True
    if p.startswith(".claude/") or p == ".claude":
        return True
    if p.startswith(".openhands/") or p == ".openhands":
        return True
    if name in {".token_usage", "token_usage", "run_instance.log", "report.json"}:
        return True
    return False


def is_scratch_path(path: str) -> bool:
    p = normalize_path(path).lower()
    name = path_basename(p)
    scratch_names = {
        "test_issue.py", "comprehensive_test.py", "manual_test.py", "quick_test.py",
        "debug.py", "debug_test.py", "repro.py", "reproduce.py", "scratch.py",
        "tmp.py", "temp.py", "check.py",
    }
    if name in scratch_names:
        return True
    if re.match(r"^(debug|repro|scratch|tmp|temp|check)[_-].*", name):
        return True
    if re.match(r".*[_-](debug|repro|scratch|tmp|temp|check)\.[^.]+$", name):
        return True
    parts = p.split("/")
    return any(part in {"tmp", "temp", "scratch", ".scratch", "debug"} for part in parts)


def is_repo_test_path(path: str) -> bool:
    p = normalize_path(path).lower()
    name = path_basename(p)
    parts = p.split("/")
    if any(part in {"test", "tests", "testing", "spec", "specs"} for part in parts[:-1]):
        return True
    return bool(
        name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".spec.js")
        or name.endswith(".test.js")
        or name.endswith(".spec.ts")
        or name.endswith(".test.ts")
        or name.endswith(".spec.tsx")
        or name.endswith(".test.tsx")
    )


def classify_path(path: str) -> str:
    if not path:
        return "empty_path"
    if is_workspace_root_path(path):
        return "workspace_root"
    if is_submission_artifact_path(path):
        return "submission_artifact"
    if is_harness_path(path):
        return "harness"
    if is_scratch_path(path):
        return "scratch"
    if is_repo_test_path(path):
        return "test"
    if path_extension(path) in SOURCE_EXTENSIONS:
        return "source"
    return "other"


def command_references_submission_artifact(command: str) -> bool:
    c = normalize_text(command).replace("\\", "/").lower()
    return (
        "_preds.json" in c
        or "model_patch" in c
        or "prediction.json" in c
        or re.search(r"(?:^|[\s;&|])(?:\./)?(?:[^\s;&|]+/)*patch\.txt(?:[\s;&|]|$)", c) is not None
        or re.search(r"(?:^|[\s;&|])(?:\./)?(?:[^\s;&|]+/)*[^/\s;&|]+\.(?:patch|diff)(?:[\s;&|]|$)", c) is not None
        or re.search(r"(?:^|[\s;&|])(?:\./)?(?:[^\s;&|]+/)*output/?(?:[\s;&|]|$)", c) is not None
        or re.search(r"\boutput/[^\s;]+", c) is not None
    )


def command_references_harness(command: str) -> bool:
    c = normalize_text(command).replace("\\", "/").lower()
    return (
        "manual.yaml" in c
        or "manual.yml" in c
        or ".claude" in c
        or ".token_usage" in c
        or "run_instance.log" in c
    )


def command_is_patch_dump_or_staging(command: str) -> bool:
    c = normalize_text(command).strip()
    git = r"\bgit(?:\s+-C\s+\S+)?"
    return bool(
        re.search(git + r"\s+diff\s+--cached\b", c, re.I)
        or re.search(git + r"\s+add\b", c, re.I)
        or re.search(git + r"\s+reset\s+HEAD\b", c, re.I)
        or re.search(git + r"\s+commit\b", c, re.I)
    )


def command_is_environment_probe_noise(command: str) -> bool:
    c = strip_harness_prompt_junk(command).strip()
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
    if re.fullmatch(ls_root + redir + pipe_suffix + fallback_echo, c, re.I):
        return True
    return False


def command_is_git_state_noise(command: str) -> bool:
    c = strip_harness_prompt_junk(command).strip()
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


def classify_command(command: str) -> str:
    cmd = normalize_text(command).strip()
    if command_references_submission_artifact(cmd):
        return "submission_artifact"
    if command_references_harness(cmd):
        return "harness"
    if command_is_patch_dump_or_staging(cmd):
        return "patch_dump_or_staging"
    if command_is_git_state_noise(cmd):
        return "diagnostic_environment_noise"
    if command_is_environment_probe_noise(cmd):
        return "diagnostic_environment_noise"
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


def text_contains_patch_or_submission(text: str) -> bool:
    return text_contains_hard_forbidden_payload(text)


def text_is_environment_listing_noise(text: str) -> bool:
    lines = [ln.strip() for ln in normalize_text(text).split("\n") if ln.strip()]
    if not lines:
        return True
    joined = "\n".join(lines).lower()
    noisy_tokens = {".claude", ".token_usage", "manual.yaml", "manual.yml", "output", "run_instance.log"}
    if len(lines) <= 3 and joined in {"testbed does not exist", "./testbed", "testbed", "."}:
        return True
    # Directory listings often contain only these harness files plus . and ...
    if len(lines) <= 20 and any(tok in joined for tok in noisy_tokens):
        repoish = any(re.search(r"\b(src|lib|sympy|django|sklearn|tests?|package\.json|pyproject\.toml)\b", ln.lower()) for ln in lines)
        if not repoish:
            return True
    return False


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
    return (
        text_is_environment_listing_noise(text)
        or text_is_missing_path_noise(text)
        or text_is_setup_failure_noise(text)
    )


def extract_primary_path(kwargs: dict[str, Any], function: str) -> tuple[str, str]:
    if function == "Glob":
        raw = kwargs.get("path") or ""
        return normalize_path_with_raw(raw) if isinstance(raw, str) and raw.strip() else ("", "")
    for key in ("file_path", "path", "notebook_path"):
        val = kwargs.get(key)
        if isinstance(val, str) and val.strip():
            return normalize_path_with_raw(val)
    return "", ""


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
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
                raise ValueError(f"{path}:{lineno}: expected JSON object")
            yield lineno, obj


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(stable_json_dumps(record) + "\n")


# ---------------------------------------------------------------------------
# ADP manifest creation
# ---------------------------------------------------------------------------


def make_issue_prompt(details: dict[str, Any], original_user_text: str = "") -> str:
    meta = details.get("issue_metadata") if isinstance(details.get("issue_metadata"), dict) else {}
    repo = normalize_text(meta.get("repo") or "")
    instance_id = normalize_text(meta.get("instance_id") or "")
    base_commit = normalize_text(meta.get("base_commit") or "")
    problem_statement = strip_harness_prompt_junk(meta.get("problem_statement") or "") or strip_harness_prompt_junk(original_user_text)
    parts = ["PRIOR ISSUE METADATA"]
    if repo:
        parts.append(f"repo: {repo}")
    if instance_id:
        parts.append(f"instance_id: {instance_id}")
    if base_commit:
        parts.append(f"base_commit: {base_commit}")
    parts.append("problem_statement:")
    parts.append(problem_statement)
    return "\n".join(parts).strip()


def item_provenance(item: dict[str, Any]) -> dict[str, Any]:
    prov = item.get("provenance")
    if isinstance(prov, dict):
        return deepcopy(prov)
    # Future-proof: accept common direct provenance fields if the converter is patched.
    out: dict[str, Any] = {}
    for key in ("raw_jsonl_line", "raw_uuid", "raw_parent_uuid", "tool_use_id", "raw_role", "raw_block_type", "raw_block_index"):
        if key in item:
            out[key] = item[key]
    return out


def nearest_preceding_action(decisions: list[dict[str, Any]], content: list[dict[str, Any]], idx: int) -> int:
    for j in range(idx - 1, -1, -1):
        if content[j].get("class_") in {"api_action", "code_action"}:
            return j
    return -1


def decide_adp_item(idx: int, item: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    base_reasons: list[str] = []
    cls = item.get("class_")

    if "reasoning_content" in item and not args.keep_reasoning_content:
        base_reasons.append("strip.reasoning_content_field_removed")

    def ret(decision: str, reasons: list[str]) -> tuple[str, list[str]]:
        return decision, base_reasons + reasons

    if cls == "text_observation":
        source = item.get("source")
        text = strip_harness_prompt_junk(item.get("content") or "")
        name = normalize_text(item.get("name") or "")
        if source == "environment":
            if not text.strip():
                return ret("drop", ["drop.empty_environment_observation"])
            if "TodoWrite" in name or name.startswith("TodoWrite:") or "Todos have been modified" in text:
                return ret("drop", ["drop.todo_observation"])
            if text_contains_hard_forbidden_payload(text):
                return ret("drop", ["drop.hard_forbidden_environment_payload"])
        if source == "user":
            return ret("keep", ["keep.user_issue_prompt"])
        return ret("keep", ["keep.text_observation"])

    if cls == "message_action":
        if args.include_final_message:
            return ret("keep", ["keep.final_message_explicitly_included"])
        return ret("drop", ["drop.final_message_default_off"])

    if cls == "api_action":
        fn = str(item.get("function") or "")
        kwargs = item.get("kwargs") if isinstance(item.get("kwargs"), dict) else {}
        kwargs_text = stable_json_dumps(kwargs)

        if fn in PLANNING_TOOLS:
            if args.include_todowrite:
                return ret("keep", ["keep.todowrite_explicitly_included"])
            return ret("drop", ["drop.todowrite"])

        if text_contains_hard_forbidden_payload(kwargs_text):
            if "model_patch" in kwargs_text or "_preds.json" in kwargs_text or "prediction.json" in kwargs_text:
                return ret("drop", ["drop.submission_payload"])
            return ret("drop", ["drop.harness_payload"])

        path, _ = extract_primary_path(kwargs, fn)
        path_class = classify_path(path)
        if path_class == "submission_artifact":
            return ret("drop", ["drop.submission_artifact_path"])
        if path_class == "harness":
            return ret("drop", ["drop.harness_path"])
        if path_class == "scratch" and fn in EDIT_TOOLS:
            return ret("drop", ["drop.scratch_file_body"])
        if path_class == "scratch" and args.drop_scratch_reads:
            return ret("drop", ["drop.scratch_file_read"])
        if fn in INSPECT_TOOLS:
            return ret("keep", [f"keep.{fn.lower()}_evidence"])
        if fn in EDIT_TOOLS:
            return ret("keep", ["keep.edit_evidence"])
        return ret("keep", ["keep.api_action"])

    if cls == "code_action":
        lang = item.get("language")
        command = strip_harness_prompt_junk(item.get("content") or "")
        if lang == "bash":
            if command_contains_patch_payload(command):
                return ret("drop", ["drop.patch_payload_command"])
            ccls = classify_command(command)
            if ccls == "submission_artifact":
                return ret("drop", ["drop.submission_artifact_command"])
            if ccls == "harness":
                return ret("drop", ["drop.harness_command"])
            if ccls == "patch_dump_or_staging":
                return ret("drop", ["drop.patch_dump_or_staging_command"])
            if ccls == "diagnostic_environment_noise":
                return ret("drop", ["drop.diagnostic_environment_noise_command"])
            if ccls == "destructive" and args.drop_destructive_commands:
                return ret("drop", ["drop.destructive_command"])
            if ccls == "setup" and args.drop_setup_commands:
                return ret("drop", ["drop.setup_command"])
            if ccls == "build_install" and args.drop_build_install_commands:
                return ret("drop", ["drop.build_install_command"])
            if ccls == "diagnostic":
                return ret("keep", ["keep.diagnostic_command"])
            if ccls == "test":
                return ret("keep", ["keep.test_command"])
            return ret("keep", ["keep.other_command"])
        return ret("keep", ["keep.code_action"])

    return ret("keep", ["keep.unknown_item_class"])

def sanitize_kept_adp_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = deepcopy(item)
    if not args.keep_reasoning_content:
        out.pop("reasoning_content", None)
    if args.drop_action_descriptions and out.get("class_") in {"api_action", "code_action", "message_action"}:
        out["description"] = None
    out = strip_harness_prompt_junk_from_value(out)
    cls = out.get("class_")
    if cls == "text_observation":
        out["content"] = strip_harness_prompt_junk(out.get("content") or "")
    elif cls in {"code_action", "message_action"}:
        out["content"] = strip_harness_prompt_junk(out.get("content") or "")
    elif cls == "api_action":
        kwargs = out.get("kwargs") if isinstance(out.get("kwargs"), dict) else {}
        out["kwargs"] = strip_harness_prompt_junk_from_value(kwargs)
    return out


def strip_reasoning_field(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return sanitize_kept_adp_item(item, args)


def build_manifest_and_stripped_traj(traj: dict[str, Any], args: argparse.Namespace, record_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    details = traj.get("details") if isinstance(traj.get("details"), dict) else {}
    content = traj.get("content") if isinstance(traj.get("content"), list) else []
    trajectory_id = normalize_text(traj.get("id") or f"record-{record_index}")

    raw_decisions: list[dict[str, Any]] = []
    stripped_content: list[dict[str, Any]] = []
    index_map: dict[int, int] = {}
    first_user_seen = False

    # First pass: action/local decisions.
    for idx, item_any in enumerate(content):
        if not isinstance(item_any, dict):
            decision = {
                "adp_content_index": idx,
                "class_": type(item_any).__name__,
                "decision": "drop",
                "reason_codes": ["drop.non_object_content_item"],
                "item_sha256": sha256_json(item_any),
                "provenance": {},
            }
        else:
            decision_value, reasons = decide_adp_item(idx, item_any, args)
            decision = {
                "adp_content_index": idx,
                "class_": normalize_text(item_any.get("class_") or ""),
                "decision": decision_value,
                "reason_codes": reasons,
                "item_sha256": sha256_json(item_any),
                "provenance": item_provenance(item_any),
            }
            if item_any.get("class_") == "code_action" and item_any.get("language") == "bash":
                decision["command_class"] = classify_command(item_any.get("content") or "")
            if item_any.get("class_") == "api_action":
                fn = normalize_text(item_any.get("function") or "")
                kwargs = item_any.get("kwargs") if isinstance(item_any.get("kwargs"), dict) else {}
                p, _ = extract_primary_path(kwargs, fn)
                if p:
                    decision["path"] = p
                    decision["path_class"] = classify_path(p)
        raw_decisions.append(decision)

    # Second pass: drop observations following dropped actions, and obvious listing noise.
    for idx, item_any in enumerate(content):
        if not isinstance(item_any, dict):
            continue
        if item_any.get("class_") == "text_observation" and item_any.get("source") == "environment":
            prev = nearest_preceding_action(raw_decisions, content, idx)
            if prev >= 0 and raw_decisions[prev]["decision"] == "drop":
                raw_decisions[idx]["decision"] = "drop"
                raw_decisions[idx]["reason_codes"] = ["drop.observation_after_dropped_action"]
            elif is_edit_tool_result_observation(content, idx):
                raw_decisions[idx]["decision"] = "drop"
                raw_decisions[idx]["reason_codes"] = ["drop.edit_tool_result_snippet"]
            elif text_is_environment_observation_noise(strip_harness_prompt_junk(item_any.get("content") or "")):
                raw_decisions[idx]["decision"] = "drop"
                raw_decisions[idx]["reason_codes"] = ["drop.environment_observation_noise"]

    for idx, item_any in enumerate(content):
        if raw_decisions[idx]["decision"] != "keep" or not isinstance(item_any, dict):
            continue
        item = sanitize_kept_adp_item(item_any, args)
        if item.get("class_") == "text_observation" and item.get("source") == "user" and args.normalize_user_prompt:
            if not first_user_seen:
                item["content"] = make_issue_prompt(details, item.get("content") or "")
                first_user_seen = True
            else:
                # Additional user turns are rare in these trajectories and often contain harness chatter.
                # Keep them only if explicitly requested.
                if not args.keep_additional_user_messages:
                    raw_decisions[idx]["decision"] = "drop"
                    raw_decisions[idx]["reason_codes"] = ["drop.additional_user_message_default_off"]
                    continue
        item["strip_source"] = {
            "original_adp_content_index": idx,
            "original_item_sha256": sha256_json(item_any),
        }
        index_map[idx] = len(stripped_content)
        stripped_content.append(item)

    stripped_details = deepcopy(details)
    for key in ADP_DETAILS_DROP_KEYS:
        stripped_details.pop(key, None)
    stripped_details = strip_harness_prompt_junk_from_value(stripped_details)
    issue_meta = stripped_details.get("issue_metadata") if isinstance(stripped_details.get("issue_metadata"), dict) else None
    if issue_meta is not None and "problem_statement" in issue_meta:
        issue_meta["problem_statement"] = strip_harness_prompt_junk(issue_meta.get("problem_statement") or "")
    stripped_details.setdefault("strip_metadata", {})
    stripped_details["strip_metadata"] = {
        "strip_policy_version": STRIP_POLICY_VERSION,
        "strip_config_hash": "",  # filled after config known
        "source_trajectory_sha256": sha256_json(traj),
        "source_content_sha256": sha256_json(content),
        "kept_content_items": len(stripped_content),
        "dropped_content_items": len(content) - len(stripped_content),
    }

    stripped_traj = {
        "schema_version": traj.get("schema_version"),
        "id": trajectory_id,
        "content": stripped_content,
        "details": stripped_details,
    }

    reason_counts = Counter(reason for d in raw_decisions for reason in d["reason_codes"])
    decision_counts = Counter(d["decision"] for d in raw_decisions)
    manifest = {
        "strip_manifest_schema_version": "1.0.0",
        "strip_policy_version": STRIP_POLICY_VERSION,
        "trajectory_id": trajectory_id,
        "source_schema_version": normalize_text(traj.get("schema_version") or ""),
        "source_file": normalize_text(details.get("source_file") or ""),
        "source_trajectory_sha256": sha256_json(traj),
        "source_content_sha256": sha256_json(content),
        "stripped_trajectory_sha256": sha256_json(stripped_traj),
        "decisions": raw_decisions,
        "index_map": {str(k): v for k, v in sorted(index_map.items())},
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "warnings": [],
    }
    if not any(d.get("provenance") for d in raw_decisions):
        manifest["warnings"].append(
            "No per-content raw provenance was found. Stripped ADP/memory are manifest-exact; raw transcript stripping uses the same policy but cannot be exact event-mask rendering."
        )
    return manifest, stripped_traj


# ---------------------------------------------------------------------------
# Memory JSONL stripping using manifest indices
# ---------------------------------------------------------------------------


def manifest_by_id(manifests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {m["trajectory_id"]: m for m in manifests}


def kept_index_set(manifest: dict[str, Any]) -> set[int]:
    return {int(d["adp_content_index"]) for d in manifest.get("decisions", []) if d.get("decision") == "keep"}


def kept_observation_index_set(manifest: dict[str, Any]) -> set[int]:
    return {
        int(d["adp_content_index"])
        for d in manifest.get("decisions", [])
        if d.get("decision") == "keep" and d.get("class_") == "text_observation"
    }


def filter_items_by_indices(items: list[Any], keep_actions: set[int], keep_obs: set[int]) -> list[Any]:
    out = []
    index_keys = ("action_index", "source_action_index", "observation_index")
    for item in items:
        if not isinstance(item, dict):
            continue
        has_index = any(isinstance(item.get(k), int) for k in index_keys)
        if not has_index:
            # Evidence arrays must be explicitly tied to ADP content. Index-less
            # entries are unsafe by default because future schema additions or
            # converter bugs could otherwise leak dropped evidence.
            continue
        ai = item.get("action_index")
        sai = item.get("source_action_index")
        oi = item.get("observation_index")
        ok = True
        if isinstance(ai, int) and ai not in keep_actions:
            ok = False
        if isinstance(sai, int) and sai not in keep_actions:
            ok = False
        if isinstance(oi, int) and oi not in keep_obs:
            ok = False
        if ok:
            out.append(strip_harness_prompt_junk_from_value(item))
    return out


def strip_memory_record(memory: dict[str, Any], manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    keep_actions = kept_index_set(manifest)
    keep_obs = kept_observation_index_set(manifest)
    out = deepcopy(memory)

    if isinstance(out.get("prior_problem"), dict):
        pp = out["prior_problem"]
        problem_statement = strip_harness_prompt_junk(pp.get("problem_statement") or "")
        pp["problem_statement"] = problem_statement
        pp["raw_user_prompt"] = problem_statement
        pp["sha256"] = sha256_text(problem_statement) if problem_statement else ""
        pp["raw_user_prompt_sha256"] = sha256_text(problem_statement) if problem_statement else ""
        pp["raw_user_prompt_truncated"] = False

    # Filter arrays with explicit source indices.
    if isinstance(out.get("search_anchors"), list):
        out["search_anchors"] = filter_items_by_indices(out["search_anchors"], keep_actions, keep_obs)

    if isinstance(out.get("commands"), dict):
        for bucket, items in list(out["commands"].items()):
            if isinstance(items, list):
                out["commands"][bucket] = filter_items_by_indices(items, keep_actions, keep_obs)

    if isinstance(out.get("observed_failures"), dict):
        for bucket, items in list(out["observed_failures"].items()):
            if isinstance(items, list):
                out["observed_failures"][bucket] = filter_items_by_indices(items, keep_actions, keep_obs)

    if isinstance(out.get("observation_evidence"), list):
        out["observation_evidence"] = filter_items_by_indices(out["observation_evidence"], keep_actions, keep_obs)

    if isinstance(out.get("edits"), dict):
        for bucket, items in list(out["edits"].items()):
            if isinstance(items, list):
                filtered = filter_items_by_indices(items, keep_actions, keep_obs)
                # Hard-remove scratch/submission buckets by default even if source index survived.
                if bucket in {"scratch_edits", "submission_artifacts"}:
                    filtered = []
                if args.edit_body_mode == "hashes-only":
                    for row in filtered:
                        if isinstance(row, dict):
                            row["old_text_excerpt"] = ""
                            row["new_text_excerpt"] = ""
                out["edits"][bucket] = filtered

    out["planning_actions"] = []
    out["reasoning"] = []
    if isinstance(out.get("prior_agent_final_message"), dict):
        out["prior_agent_final_message"] = {
            "text": "",
            "sha256": out["prior_agent_final_message"].get("sha256", ""),
            "truncated": False,
            "included_in_rendered_text": False,
            "stripped": True,
        }

    # Path stores have only first_action_index, so filter conservatively.
    if isinstance(out.get("files"), dict):
        for key in ("inspected", "edited"):
            rows = out["files"].get(key)
            if isinstance(rows, list):
                kept_rows = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    path = normalize_path(row.get("path") or "")
                    pcls = classify_path(path)
                    if pcls in {"submission_artifact", "harness", "scratch", "workspace_root"} or is_workspace_root_path(path):
                        continue
                    first = row.get("first_action_index")
                    if isinstance(first, int) and first not in keep_actions:
                        continue
                    kept_rows.append(row)
                out["files"][key] = kept_rows

    out["rendered_text"] = render_stripped_memory_text(out)
    out.setdefault("strip_metadata", {})
    out["strip_metadata"] = {
        "strip_policy_version": STRIP_POLICY_VERSION,
        "source_manifest_sha256": sha256_json(manifest),
        "note": "This memory was filtered by source ADP content indices. Prefer generating memory directly from stripped ADP for strongest parity.",
    }
    return out


def render_stripped_memory_text(memory: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("PRIOR EXPERIENCE MEMORY")
    lines.append("This is deterministic evidence from a previous stripped trajectory. Use as context only.")
    src = memory.get("source_metadata") if isinstance(memory.get("source_metadata"), dict) else {}
    lines.extend(["", "SOURCE", "------"])
    lines.append(f"- trajectory_id: {memory.get('source_trajectory_id', '')}")
    lines.append(f"- repo: {src.get('repo', '')}")
    lines.append(f"- instance_id: {src.get('instance_id', '')}")
    lines.append(f"- base_commit: {src.get('base_commit', '')}")

    problem = memory.get("prior_problem") if isinstance(memory.get("prior_problem"), dict) else {}
    lines.extend(["", "PRIOR PROBLEM", "-------------"])
    lines.append(strip_harness_prompt_junk(problem.get("problem_statement") or "(not found)"))

    files = memory.get("files") if isinstance(memory.get("files"), dict) else {}
    inspected = files.get("inspected") if isinstance(files.get("inspected"), list) else []
    lines.extend(["", "FILES INSPECTED", "---------------"])
    if inspected:
        for row in inspected:
            if isinstance(row, dict):
                lines.append(f"- {row.get('path', '')}")
    else:
        lines.append("(none)")

    anchors = memory.get("search_anchors") if isinstance(memory.get("search_anchors"), list) else []
    lines.extend(["", "SEARCH ANCHORS", "--------------"])
    if anchors:
        for a in anchors:
            if not isinstance(a, dict):
                continue
            parts = [f"- [{a.get('action_index', '')}] {a.get('tool', '')}"]
            if a.get("pattern"):
                parts.append(f"pattern={a.get('pattern')!r}")
            if a.get("path"):
                parts.append(f"path={a.get('path')!r}")
            if a.get("glob"):
                parts.append(f"glob={a.get('glob')!r}")
            lines.append(" ".join(parts))
    else:
        lines.append("(none)")

    commands = memory.get("commands") if isinstance(memory.get("commands"), dict) else {}
    lines.extend(["", "TEST / DIAGNOSTIC / OTHER COMMANDS", "----------------------------------"])
    rendered_any = False
    for bucket in ("test_commands", "diagnostic_commands", "other_commands"):
        for c in commands.get(bucket, []) if isinstance(commands.get(bucket), list) else []:
            if isinstance(c, dict):
                lines.append(f"- [{c.get('command_class', bucket)}] {strip_harness_prompt_junk(c.get('command') or '')}")
                rendered_any = True
    if not rendered_any:
        lines.append("(none)")

    failures = memory.get("observed_failures") if isinstance(memory.get("observed_failures"), dict) else {}
    lines.extend(["", "OBSERVED TEST / RUNTIME FAILURES", "--------------------------------"])
    failure_any = False
    for bucket in ("test_failures", "runtime_failures", "other_failures"):
        for f in failures.get(bucket, []) if isinstance(failures.get(bucket), list) else []:
            if isinstance(f, dict):
                lines.append(f"- {strip_harness_prompt_junk(f.get('line', ''))}")
                failure_any = True
    if not failure_any:
        lines.append("(none)")

    evidence = memory.get("observation_evidence") if isinstance(memory.get("observation_evidence"), list) else []
    lines.extend(["", "RELEVANT OBSERVATION EXCERPTS", "-----------------------------"])
    if evidence:
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            lines.append(f"- [{ev.get('kind', 'observation')} after action #{ev.get('source_action_index', '')}]")
            for line in normalize_text(ev.get("excerpt") or "").split("\n"):
                lines.append(f"  {strip_harness_prompt_junk(line)}")
    else:
        lines.append("(none)")

    edits = memory.get("edits") if isinstance(memory.get("edits"), dict) else {}
    lines.extend(["", "SOURCE EDIT EVIDENCE", "--------------------"])
    edit_any = False
    for bucket in ("source_edits", "test_edits", "other_edits"):
        for e in edits.get(bucket, []) if isinstance(edits.get(bucket), list) else []:
            if not isinstance(e, dict):
                continue
            lines.append(f"- {strip_harness_prompt_junk(e.get('path', ''))}")
            if e.get("old_text_sha256"):
                lines.append(f"  old_text_sha256: {e.get('old_text_sha256')}")
            if e.get("new_text_sha256"):
                lines.append(f"  new_text_sha256: {e.get('new_text_sha256')}")
            if e.get("old_text_excerpt"):
                lines.append("  old_text_excerpt:")
                for line in normalize_text(e.get("old_text_excerpt") or "").split("\n"):
                    lines.append(f"    {strip_harness_prompt_junk(line)}")
            if e.get("new_text_excerpt"):
                lines.append("  new_text_excerpt:")
                for line in normalize_text(e.get("new_text_excerpt") or "").split("\n"):
                    lines.append(f"    {strip_harness_prompt_junk(line)}")
            edit_any = True
    if not edit_any:
        lines.append("(none)")

    lines.extend(["", "INJECTION WARNING", "-----------------"])
    lines.append("This memory is from a prior related issue. Use it as evidence only. Do not copy patches blindly.")
    return strip_harness_prompt_junk("\n".join(lines))


def edit_kwargs_hashes_only(kwargs: dict[str, Any], function: str) -> dict[str, Any]:
    out = deepcopy(kwargs)

    def replace_pair(container: dict[str, Any]) -> None:
        old_raw = normalize_text(container.pop("old_string", ""))
        new_raw = normalize_text(container.pop("new_string", ""))
        if old_raw:
            container["old_text_sha256"] = sha256_text(old_raw)
        if new_raw:
            container["new_text_sha256"] = sha256_text(new_raw)

    if function == "Edit":
        replace_pair(out)
    elif function == "Write":
        content_raw = normalize_text(out.pop("content", ""))
        if content_raw:
            out["content_sha256"] = sha256_text(content_raw)
    elif function == "MultiEdit":
        edits = out.get("edits")
        if isinstance(edits, list):
            new_edits = []
            for sub in edits:
                if isinstance(sub, dict):
                    sub_copy = dict(sub)
                    replace_pair(sub_copy)
                    new_edits.append(sub_copy)
            out["edits"] = new_edits
    return out


def safe_filename_component(value: str) -> str:
    value = normalize_text(value).strip() or "unknown"
    value = re.sub(r"[/\\:*?\"<>|\s]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    return value[:120] or "unknown"


# ---------------------------------------------------------------------------
# Rendered/capped ADP injection output
# ---------------------------------------------------------------------------


def render_adp_action_payload(item: dict[str, Any], args: argparse.Namespace) -> tuple[str, bool]:
    cls = item.get("class_")
    if cls == "code_action":
        return truncate_text(strip_harness_prompt_junk(item.get("content") or ""), args.max_adp_action_chars)
    if cls == "api_action":
        fn = normalize_text(item.get("function") or "")
        kwargs = item.get("kwargs") if isinstance(item.get("kwargs"), dict) else {}
        if args.edit_body_mode == "hashes-only" and fn in EDIT_TOOLS:
            kwargs = edit_kwargs_hashes_only(kwargs, fn)
        payload = stable_json_dumps(strip_harness_prompt_junk_from_value(kwargs), pretty=True)
        return truncate_text(payload, args.max_adp_action_chars)
    if cls == "message_action":
        return truncate_text(strip_harness_prompt_junk(item.get("content") or ""), args.max_adp_action_chars)
    return truncate_text(stable_json_dumps(strip_harness_prompt_junk_from_value(item), pretty=True), args.max_adp_action_chars)


def render_stripped_adp_injection_text(traj: dict[str, Any], args: argparse.Namespace) -> str:
    lines: list[str] = []
    details = traj.get("details") if isinstance(traj.get("details"), dict) else {}
    meta = details.get("issue_metadata") if isinstance(details.get("issue_metadata"), dict) else {}
    content = traj.get("content") if isinstance(traj.get("content"), list) else []
    lines.append("STRIPPED ADP TRAJECTORY")
    lines.append("This is the capped/rendered ADP representation for injection. Full stripped_adp.audit.jsonl is audit-only.")
    lines.extend(["", "SOURCE", "------"])
    lines.append(f"- trajectory_id: {traj.get('id', '')}")
    lines.append(f"- repo: {meta.get('repo', '')}")
    lines.append(f"- instance_id: {meta.get('instance_id', '')}")
    lines.append(f"- base_commit: {meta.get('base_commit', '')}")

    for idx, item_any in enumerate(content):
        if not isinstance(item_any, dict):
            continue
        item = item_any
        cls = normalize_text(item.get("class_") or "")
        strip_source = item.get("strip_source") if isinstance(item.get("strip_source"), dict) else {}
        original_idx = strip_source.get("original_adp_content_index", "")
        idx_label = f"current={idx} original={original_idx}" if original_idx != "" else f"current={idx}"
        if cls == "text_observation":
            src = normalize_text(item.get("source") or "")
            text = strip_harness_prompt_junk(item.get("content") or "")
            max_chars = args.max_adp_observation_chars
            excerpt, truncated = truncate_text(text, max_chars)
            lines.extend(["", f"[ADP OBSERVATION {idx_label} source={src}]"])
            if item.get("name"):
                lines.append(f"name: {strip_harness_prompt_junk(item.get('name'))}")
            lines.append("excerpt:")
            lines.append(excerpt)
            lines.append(f"sha256: {sha256_text(text)}")
            lines.append(f"truncated: {str(truncated).lower()}")
        elif cls in {"api_action", "code_action", "message_action"}:
            if cls == "api_action":
                label = f"api_action {item.get('function', '')}"
            elif cls == "code_action":
                label = f"code_action {item.get('language', '')}"
            else:
                label = "message_action"
            payload, truncated = render_adp_action_payload(item, args)
            lines.extend(["", f"[ADP ACTION {idx_label}: {label}]"])
            if cls == "api_action":
                kwargs = item.get("kwargs") if isinstance(item.get("kwargs"), dict) else {}
                path, _raw = extract_primary_path(kwargs, normalize_text(item.get("function") or ""))
                if path:
                    lines.append(f"path: {strip_harness_prompt_junk(path)}")
            lines.append("payload:")
            lines.append(payload)
            lines.append(f"sha256: {sha256_json(item)}")
            lines.append(f"truncated: {str(truncated).lower()}")
        else:
            payload, truncated = truncate_text(stable_json_dumps(strip_harness_prompt_junk_from_value(item), pretty=True), args.max_adp_action_chars)
            lines.extend(["", f"[ADP ITEM {idx_label}: {cls}]"])
            lines.append(payload)
            lines.append(f"truncated: {str(truncated).lower()}")
    return strip_harness_prompt_junk("\n".join(lines)).rstrip() + "\n"


def write_rendered_adp_injections(stripped_trajs: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> list[Path]:
    rendered_dir = out_dir / "stripped_adp_rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, traj in enumerate(stripped_trajs, start=1):
        safe = safe_filename_component(normalize_text(traj.get("id") or f"trajectory-{idx}"))
        path = rendered_dir / f"{idx:06d}_{safe}.txt"
        path.write_text(render_stripped_adp_injection_text(traj, args), encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def sanitize_adp_item_for_injection(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Return an injection-safe ADP item with capped text and hashed edit bodies."""
    out = sanitize_kept_adp_item(item, args)
    cls = out.get("class_")
    if cls == "text_observation":
        raw = strip_harness_prompt_junk(out.get("content") or "")
        excerpt, truncated = truncate_text(raw, args.max_adp_observation_chars)
        out["content"] = excerpt
        out["content_sha256"] = sha256_text(raw)
        out["content_truncated"] = truncated
    elif cls == "code_action":
        raw = strip_harness_prompt_junk(out.get("content") or "")
        excerpt, truncated = truncate_text(raw, args.max_adp_action_chars)
        out["content"] = excerpt
        out["content_sha256"] = sha256_text(raw)
        out["content_truncated"] = truncated
    elif cls == "message_action":
        raw = strip_harness_prompt_junk(out.get("content") or "")
        excerpt, truncated = truncate_text(raw, args.max_adp_action_chars)
        out["content"] = excerpt
        out["content_sha256"] = sha256_text(raw)
        out["content_truncated"] = truncated
    elif cls == "api_action":
        fn = normalize_text(out.get("function") or "")
        kwargs = out.get("kwargs") if isinstance(out.get("kwargs"), dict) else {}
        if fn in EDIT_TOOLS:
            out["kwargs"] = edit_kwargs_hashes_only(kwargs, fn)
            out["edit_body_mode"] = "hashes-only"
        else:
            payload = stable_json_dumps(strip_harness_prompt_junk_from_value(kwargs), pretty=True)
            if len(payload) > args.max_adp_action_chars:
                out["kwargs_sha256"] = sha256_text(payload)
                out["kwargs_truncated"] = True
                out["kwargs"] = {"truncated_payload_sha256": out["kwargs_sha256"]}
            else:
                out["kwargs"] = strip_harness_prompt_junk_from_value(kwargs)
                out["kwargs_truncated"] = False
    return strip_harness_prompt_junk_from_value(out)


def make_injection_safe_adp_traj(traj: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = deepcopy(traj)
    content = traj.get("content") if isinstance(traj.get("content"), list) else []
    out["content"] = [sanitize_adp_item_for_injection(item, args) for item in content if isinstance(item, dict)]
    details = out.get("details") if isinstance(out.get("details"), dict) else {}
    details = deepcopy(details)
    details.setdefault("strip_metadata", {})
    if isinstance(details["strip_metadata"], dict):
        details["strip_metadata"]["injection_safe"] = True
        details["strip_metadata"]["edit_body_mode"] = "hashes-only"
        details["strip_metadata"]["source_audit_jsonl"] = "stripped_adp.audit.jsonl"
    out["details"] = details
    return out


# ---------------------------------------------------------------------------
# Raw Claude JSONL stripped transcript rendering
# ---------------------------------------------------------------------------


def as_blocks(message_content: Any) -> list[dict[str, Any]]:
    if isinstance(message_content, str):
        return [{"type": "text", "text": message_content}]
    if isinstance(message_content, list):
        return [b if isinstance(b, dict) else {"type": "unknown", "value": b} for b in message_content]
    if message_content is None:
        return []
    return [{"type": "unknown", "value": message_content}]


def stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
            elif isinstance(item, dict) and "content" in item:
                pieces.append(stringify_content(item.get("content")))
            else:
                pieces.append(stable_json_dumps(item))
        return "\n".join(pieces)
    if isinstance(value, dict):
        return stable_json_dumps(value)
    return str(value)


def raw_tool_decision(tool_name: str, tool_input: Any, args: argparse.Namespace) -> tuple[str, list[str], str]:
    inp = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
    if tool_name == "TodoWrite":
        if args.include_todowrite:
            return "keep", ["keep.todowrite_explicitly_included"], "planning"
        return "drop", ["drop.todowrite"], "planning"
    if tool_name == "Bash":
        command = normalize_text(inp.get("command") or "")
        if command_contains_patch_payload(command):
            return "drop", ["drop.patch_payload_command"], "other"
        ccls = classify_command(command)
        if ccls == "submission_artifact":
            return "drop", ["drop.submission_artifact_command"], ccls
        if ccls == "harness":
            return "drop", ["drop.harness_command"], ccls
        if ccls == "patch_dump_or_staging":
            return "drop", ["drop.patch_dump_or_staging_command"], ccls
        if ccls == "diagnostic_environment_noise":
            return "drop", ["drop.diagnostic_environment_noise_command"], ccls
        if ccls == "destructive" and args.drop_destructive_commands:
            return "drop", ["drop.destructive_command"], ccls
        if ccls == "setup" and args.drop_setup_commands:
            return "drop", ["drop.setup_command"], ccls
        if ccls == "build_install" and args.drop_build_install_commands:
            return "drop", ["drop.build_install_command"], ccls
        return "keep", [f"keep.{ccls}_command"], ccls
    if tool_name in INSPECT_TOOLS or tool_name in EDIT_TOOLS:
        payload_text = stable_json_dumps(inp)
        if text_contains_hard_forbidden_payload(payload_text):
            if "model_patch" in payload_text or "_preds.json" in payload_text or "prediction.json" in payload_text:
                return "drop", ["drop.submission_payload"], "other"
            return "drop", ["drop.harness_payload"], "other"
        p, _ = extract_primary_path(inp, tool_name)
        pcls = classify_path(p)
        if pcls == "submission_artifact":
            return "drop", ["drop.submission_artifact_path"], pcls
        if pcls == "harness":
            return "drop", ["drop.harness_path"], pcls
        if pcls == "scratch" and tool_name in EDIT_TOOLS:
            return "drop", ["drop.scratch_file_body"], pcls
        if pcls == "scratch" and args.drop_scratch_reads:
            return "drop", ["drop.scratch_file_read"], pcls
        return "keep", [f"keep.{tool_name.lower()}"], pcls
    return "keep", ["keep.raw_tool_use"], "tool"


def extract_issue_metadata_from_prompt(first_user_text: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for key in ("instance_id", "repo", "base_commit"):
        m = re.search(rf"(?m)^\s*{re.escape(key)}:\s*(.+?)\s*$", first_user_text)
        if m:
            meta[key] = m.group(1).strip()
    ps = re.search(r"(?ms)^\s*problem_statement:\s*(.*)$", first_user_text)
    if ps:
        meta["problem_statement"] = ps.group(1).strip()
    return meta


def render_raw_issue_prompt(text: str) -> str:
    meta = extract_issue_metadata_from_prompt(text)
    if not meta:
        return strip_harness_prompt_junk(text)
    parts = ["PRIOR ISSUE METADATA"]
    for key in ("repo", "instance_id", "base_commit"):
        if meta.get(key):
            parts.append(f"{key}: {meta[key]}")
    parts.append("problem_statement:")
    parts.append(strip_harness_prompt_junk(meta.get("problem_statement", "")))
    return "\n".join(parts).strip()


def render_raw_transcript_policy(raw_path: Path, args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    lines: list[str] = []
    decisions: list[dict[str, Any]] = []
    tool_decisions: dict[str, str] = {}
    tool_reason_codes: dict[str, list[str]] = {}
    tool_names: dict[str, str] = {}
    issue_user_rendered = False

    lines.append("STRIPPED RAW CLAUDE CODE TRAJECTORY")
    lines.append("This transcript was rendered with the same deterministic junk policy as ADP stripping.")
    lines.append("")

    for lineno, rec in iter_jsonl(raw_path):
        rtype = rec.get("type")
        if rtype in {"summary", "file-history-snapshot"}:
            decisions.append({"raw_jsonl_line": lineno, "decision": "drop", "reason_codes": [f"drop.raw_{rtype}"]})
            continue
        msg = rec.get("message") if isinstance(rec.get("message"), dict) else None
        if not msg:
            decisions.append({"raw_jsonl_line": lineno, "decision": "drop", "reason_codes": ["drop.raw_non_message_record"]})
            continue
        role = msg.get("role") or rec.get("type")
        blocks = as_blocks(msg.get("content"))
        block_any_kept = False

        for bi, block in enumerate(blocks):
            btype = block.get("type")
            if btype == "thinking":
                if args.keep_reasoning_content:
                    text = strip_harness_prompt_junk(stringify_content(block.get("thinking")))
                    lines.append(f"[RAW line {lineno} THINKING]")
                    lines.append(text)
                    block_any_kept = True
                else:
                    decisions.append({"raw_jsonl_line": lineno, "block_index": bi, "decision": "drop", "reason_codes": ["drop.reasoning_content"]})
                continue

            if role == "user" and btype == "text":
                text = strip_harness_prompt_junk(block.get("text") or "")
                if issue_user_rendered and not args.keep_additional_user_messages:
                    decisions.append({"raw_jsonl_line": lineno, "block_index": bi, "decision": "drop", "reason_codes": ["drop.additional_user_message_default_off"]})
                    continue
                lines.append(f"[RAW line {lineno} USER]")
                if args.normalize_user_prompt and not issue_user_rendered:
                    # Raw alone does not know extracted issue metadata unless the prompt contains it.
                    lines.append(render_raw_issue_prompt(text))
                    issue_user_rendered = True
                else:
                    lines.append(text)
                block_any_kept = True
                continue

            if role == "assistant" and btype == "text":
                if args.include_final_message:
                    text = strip_harness_prompt_junk(block.get("text") or "")
                    lines.append(f"[RAW line {lineno} ASSISTANT TEXT]")
                    lines.append(text)
                    block_any_kept = True
                else:
                    decisions.append({"raw_jsonl_line": lineno, "block_index": bi, "decision": "drop", "reason_codes": ["drop.assistant_text_default_off"]})
                continue

            if role == "assistant" and btype == "tool_use":
                tool_name = normalize_text(block.get("name") or "")
                tool_id = normalize_text(block.get("id") or "")
                tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
                decision, reasons, label = raw_tool_decision(tool_name, tool_input, args)
                if tool_id:
                    tool_decisions[tool_id] = decision
                    tool_reason_codes[tool_id] = reasons
                    tool_names[tool_id] = tool_name
                decisions.append({"raw_jsonl_line": lineno, "block_index": bi, "tool_use_id": tool_id, "decision": decision, "reason_codes": reasons})
                if decision == "keep":
                    lines.append(f"[RAW line {lineno} TOOL {tool_name} {label}]")
                    if tool_name == "Bash":
                        excerpt, truncated = truncate_text(strip_harness_prompt_junk(tool_input.get("command") or ""), args.max_adp_action_chars)
                        lines.append(excerpt)
                        if truncated:
                            lines.append("[... truncated ...]")
                    else:
                        # Drop Claude's description key and hash edit bodies for injection safety.
                        cleaned = {k: v for k, v in tool_input.items() if k != "description"}
                        if args.edit_body_mode == "hashes-only" and tool_name in EDIT_TOOLS:
                            cleaned = edit_kwargs_hashes_only(cleaned, tool_name)
                        payload = stable_json_dumps(strip_harness_prompt_junk_from_value(cleaned), pretty=True)
                        excerpt, truncated = truncate_text(payload, args.max_adp_action_chars)
                        lines.append(excerpt)
                        if truncated:
                            lines.append("[... truncated ...]")
                    block_any_kept = True
                continue

            if role == "user" and btype == "tool_result":
                tool_id = normalize_text(block.get("tool_use_id") or "")
                parent_decision = tool_decisions.get(tool_id, "keep")
                parent_tool_name = tool_names.get(tool_id, "")
                text = strip_harness_prompt_junk(stringify_content(block.get("content")))
                if parent_decision == "drop":
                    decisions.append({"raw_jsonl_line": lineno, "block_index": bi, "tool_use_id": tool_id, "decision": "drop", "reason_codes": ["drop.raw_tool_result_after_dropped_tool"]})
                    continue
                if parent_tool_name in EDIT_TOOLS:
                    decisions.append({"raw_jsonl_line": lineno, "block_index": bi, "tool_use_id": tool_id, "decision": "drop", "reason_codes": ["drop.edit_tool_result_snippet"]})
                    continue
                if not text.strip():
                    decisions.append({"raw_jsonl_line": lineno, "block_index": bi, "tool_use_id": tool_id, "decision": "drop", "reason_codes": ["drop.empty_tool_result"]})
                    continue
                if text_contains_hard_forbidden_payload(text):
                    decisions.append({"raw_jsonl_line": lineno, "block_index": bi, "tool_use_id": tool_id, "decision": "drop", "reason_codes": ["drop.hard_forbidden_tool_result"]})
                    continue
                if text_is_environment_observation_noise(text):
                    decisions.append({"raw_jsonl_line": lineno, "block_index": bi, "tool_use_id": tool_id, "decision": "drop", "reason_codes": ["drop.environment_observation_noise"]})
                    continue
                excerpt, truncated = truncate_text(strip_harness_prompt_junk(text), args.max_raw_tool_result_chars)
                lines.append(f"[RAW line {lineno} TOOL RESULT {tool_id}]")
                lines.append(excerpt)
                if truncated:
                    lines.append("[... truncated ...]")
                block_any_kept = True
                decisions.append({"raw_jsonl_line": lineno, "block_index": bi, "tool_use_id": tool_id, "decision": "keep", "reason_codes": ["keep.raw_tool_result"]})
                continue

            text = strip_harness_prompt_junk(stringify_content(block))
            if text.strip():
                lines.append(f"[RAW line {lineno} {role} {btype}]")
                lines.append(text)
                block_any_kept = True

        if not block_any_kept:
            # Do not duplicate if per-block decisions already exist; this is only a record-level trace.
            pass

    manifest = {
        "raw_file": raw_path.name,
        "raw_file_sha256": sha256_text(raw_path.read_text(encoding="utf-8")),
        "decisions": decisions,
        "decision_counts": dict(sorted(Counter(d.get("decision", "") for d in decisions).items())),
        "reason_counts": dict(sorted(Counter(r for d in decisions for r in d.get("reason_codes", [])).items())),
        "warnings": [
            "Raw transcript was stripped by policy, not by exact ADP content index mask. Add Raw->ADP provenance for exact parity."
        ],
    }
    return "\n".join(lines).rstrip() + "\n", manifest


def read_raw_records_for_chain(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, obj in iter_jsonl(path):
        obj = deepcopy(obj)
        obj["_jsonl_lineno"] = lineno
        records.append(obj)
    return records


def choose_leaf_uuid(records: list[dict[str, Any]]) -> str | None:
    uuid_to_record = {str(r.get("uuid")): r for r in records if r.get("uuid")}
    for r in records:
        if r.get("type") == "summary" and r.get("leafUuid") in uuid_to_record:
            return str(r["leafUuid"])
    for r in reversed(records):
        if r.get("uuid"):
            return str(r["uuid"])
    return None


def reconstruct_raw_message_chain(records: list[dict[str, Any]], use_parent_chain: bool = True) -> list[dict[str, Any]]:
    message_records = [r for r in records if r.get("uuid") and isinstance(r.get("message"), dict)]
    if not use_parent_chain:
        return message_records
    uuid_to_record = {str(r["uuid"]): r for r in message_records}
    leaf = choose_leaf_uuid(records)
    if not leaf or leaf not in uuid_to_record:
        return message_records
    chain_reversed: list[dict[str, Any]] = []
    seen: set[str] = set()
    cur: str | None = leaf
    while cur:
        if cur in seen or cur not in uuid_to_record:
            break
        seen.add(cur)
        rec = uuid_to_record[cur]
        chain_reversed.append(rec)
        parent = rec.get("parentUuid")
        cur = str(parent) if parent else None
    chain = list(reversed(chain_reversed))
    if len(chain) < max(1, len(message_records) // 2):
        return message_records
    return chain


def manifest_has_raw_provenance(manifest: dict[str, Any] | None) -> bool:
    if not manifest:
        return False
    for d in manifest.get("decisions", []):
        if isinstance(d, dict) and isinstance(d.get("provenance"), dict) and d["provenance"].get("raw_jsonl_line"):
            return True
    return False


def kept_decision_matches_raw_block(
    decision: dict[str, Any],
    *,
    lineno: int,
    role: str,
    block_type: str,
    block_index: int,
    tool_use_id: str = "",
) -> bool:
    if decision.get("decision") != "keep":
        return False
    prov = decision.get("provenance") if isinstance(decision.get("provenance"), dict) else {}
    if prov.get("raw_jsonl_line") != lineno:
        return False
    if prov.get("raw_role") and str(prov.get("raw_role")) != role:
        return False
    if prov.get("raw_block_type") and str(prov.get("raw_block_type")) != block_type:
        return False
    prov_tool_id = normalize_text(prov.get("tool_use_id") or "")
    if tool_use_id and prov_tool_id and prov_tool_id != tool_use_id:
        return False
    if tool_use_id and block_type in {"tool_use", "tool_result"} and prov_tool_id != tool_use_id:
        return False
    prov_block_index = prov.get("raw_block_index")
    if isinstance(prov_block_index, int) and prov_block_index != block_index:
        return False
    return True


def raw_block_is_kept_by_manifest(
    manifest: dict[str, Any],
    *,
    lineno: int,
    role: str,
    block_type: str,
    block_index: int,
    tool_use_id: str = "",
) -> tuple[bool, list[dict[str, Any]]]:
    matches = [
        d
        for d in manifest.get("decisions", [])
        if isinstance(d, dict)
        and kept_decision_matches_raw_block(
            d,
            lineno=lineno,
            role=role,
            block_type=block_type,
            block_index=block_index,
            tool_use_id=tool_use_id,
        )
    ]
    return bool(matches), matches


def render_raw_transcript_manifest_exact(raw_path: Path, args: argparse.Namespace, manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    records = read_raw_records_for_chain(raw_path)
    chain = reconstruct_raw_message_chain(records, use_parent_chain=True)
    lines: list[str] = [
        "STRIPPED RAW CLAUDE CODE TRAJECTORY",
        "This transcript was rendered from original raw JSONL using the canonical ADP strip manifest.",
        "",
    ]
    decisions: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}
    issue_user_rendered = False

    for rec in chain:
        lineno = int(rec.get("_jsonl_lineno") or 0)
        msg = rec.get("message") if isinstance(rec.get("message"), dict) else None
        if not msg:
            continue
        role = str(msg.get("role") or rec.get("type") or "")
        blocks = as_blocks(msg.get("content"))
        for bi, block in enumerate(blocks):
            btype = str(block.get("type") or "unknown")
            tool_id = normalize_text(block.get("id") or block.get("tool_use_id") or "")
            kept, matched = raw_block_is_kept_by_manifest(
                manifest,
                lineno=lineno,
                role=role,
                block_type=btype,
                block_index=bi,
                tool_use_id=tool_id,
            )
            if not kept:
                decisions.append({
                    "raw_jsonl_line": lineno,
                    "block_index": bi,
                    "raw_block_type": btype,
                    "tool_use_id": tool_id,
                    "decision": "drop",
                    "reason_codes": ["drop.raw_block_not_in_kept_adp_manifest"],
                })
                continue

            adp_indices = [m.get("adp_content_index") for m in matched]
            decisions.append({
                "raw_jsonl_line": lineno,
                "block_index": bi,
                "raw_block_type": btype,
                "tool_use_id": tool_id,
                "decision": "keep",
                "adp_content_indices": adp_indices,
                "reason_codes": ["keep.raw_block_from_kept_adp_manifest"],
            })

            if btype == "thinking":
                if args.keep_reasoning_content:
                    text = strip_harness_prompt_junk(stringify_content(block.get("thinking")))
                    lines.append(f"[RAW line {lineno} THINKING]")
                    lines.append(text)
                continue
            if role == "user" and btype == "text":
                text = strip_harness_prompt_junk(block.get("text") or "")
                lines.append(f"[RAW line {lineno} USER]")
                if args.normalize_user_prompt and not issue_user_rendered:
                    lines.append(render_raw_issue_prompt(text))
                    issue_user_rendered = True
                else:
                    lines.append(text)
            elif role == "assistant" and btype == "text":
                text = strip_harness_prompt_junk(block.get("text") or "")
                lines.append(f"[RAW line {lineno} ASSISTANT TEXT]")
                lines.append(text)
            elif role == "assistant" and btype == "tool_use":
                tool_name = normalize_text(block.get("name") or "")
                tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
                if tool_id:
                    tool_names[tool_id] = tool_name
                lines.append(f"[RAW line {lineno} TOOL {tool_name}]")
                if tool_name == "Bash":
                    excerpt, truncated = truncate_text(strip_harness_prompt_junk(tool_input.get("command") or ""), args.max_adp_action_chars)
                    lines.append(excerpt)
                    if truncated:
                        lines.append("[... truncated ...]")
                else:
                    cleaned = {k: v for k, v in tool_input.items() if k != "description"}
                    if args.edit_body_mode == "hashes-only" and tool_name in EDIT_TOOLS:
                        cleaned = edit_kwargs_hashes_only(cleaned, tool_name)
                    payload = stable_json_dumps(strip_harness_prompt_junk_from_value(cleaned), pretty=True)
                    excerpt, truncated = truncate_text(payload, args.max_adp_action_chars)
                    lines.append(excerpt)
                    if truncated:
                        lines.append("[... truncated ...]")
            elif role == "user" and btype == "tool_result":
                if tool_names.get(tool_id, "") in EDIT_TOOLS:
                    continue
                text = strip_harness_prompt_junk(stringify_content(block.get("content")))
                excerpt, truncated = truncate_text(text, args.max_raw_tool_result_chars)
                lines.append(f"[RAW line {lineno} TOOL RESULT {tool_id}]")
                lines.append(excerpt)
                if truncated:
                    lines.append("[... truncated ...]")
            else:
                text = strip_harness_prompt_junk(stringify_content(block))
                if text.strip():
                    lines.append(f"[RAW line {lineno} {role} {btype}]")
                    lines.append(text)

    raw_manifest = {
        "raw_file": raw_path.name,
        "raw_file_sha256": sha256_text(raw_path.read_text(encoding="utf-8")),
        "trajectory_id": manifest.get("trajectory_id", ""),
        "source_manifest_sha256": sha256_json(manifest),
        "render_mode": "manifest_exact",
        "decisions": decisions,
        "decision_counts": dict(sorted(Counter(d.get("decision", "") for d in decisions).items())),
        "reason_counts": dict(sorted(Counter(r for d in decisions for r in d.get("reason_codes", [])).items())),
        "warnings": [],
    }
    return strip_harness_prompt_junk("\n".join(lines)).rstrip() + "\n", raw_manifest


def render_raw_transcript(raw_path: Path, args: argparse.Namespace, manifest: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    if manifest_has_raw_provenance(manifest):
        return render_raw_transcript_manifest_exact(raw_path, args, manifest)  # type: ignore[arg-type]
    transcript, raw_manifest = render_raw_transcript_policy(raw_path, args)
    raw_manifest["render_mode"] = "policy_fallback_no_provenance"
    return transcript, raw_manifest


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def forbidden_patterns() -> list[tuple[str, re.Pattern[str]]]:
    git = r"\bgit(?:\s+-C\s+\S+)?"
    return [
        ("reasoning_content", re.compile(r"reasoning_content", re.I)),
        ("TodoWrite", re.compile(r"TodoWrite", re.I)),
        ("preds_json", re.compile(r"_preds\.json|\bpreds\.json\b|prediction\.json", re.I)),
        ("model_patch", re.compile(r"model_patch", re.I)),
        ("manual_yaml", re.compile(r"manual\.ya?ml", re.I)),
        ("agent_summary", re.compile(r"Agent-Summary:", re.I)),
        ("agent_session_paths", re.compile(r"Agent-Session-Paths:", re.I)),
        ("realign_sessions", re.compile(r"\.realign/sessions/", re.I)),
        ("git_clone", re.compile(r"\bgit\s+clone\b", re.I)),
        ("git_add", re.compile(git + r"\s+add\b", re.I)),
        ("git_diff_cached", re.compile(git + r"\s+diff\s+--cached\b", re.I)),
        ("git_reset_head", re.compile(git + r"\s+reset\s+HEAD\b", re.I)),
        ("git_commit", re.compile(git + r"\s+commit\b", re.I)),
        ("pip_install", re.compile(r"\bpip\s+install\b|python\s+-m\s+pip\s+install", re.I)),
        ("rm_rf", re.compile(r"\brm\s+-[A-Za-z]*rf|\brm\s+-r\s+-f", re.I)),
        ("diff_git", re.compile(r"diff --git ", re.I)),
        ("patch_header_old", re.compile(r"^--- a/", re.I | re.M)),
        ("patch_header_new", re.compile(r"^\+\+\+ b/", re.I | re.M)),
        ("old_string_key", re.compile(r"__KEY__:old_string\b", re.I)),
        ("new_string_key", re.compile(r"__KEY__:new_string\b", re.I)),
        ("claude_dir", re.compile(r"\.claude", re.I)),
        ("token_usage", re.compile(r"\.token_usage", re.I)),
        ("run_instance_log", re.compile(r"run_instance\.log", re.I)),
        ("system_reminder_open", re.compile(r"<system-reminder>", re.I)),
        ("system_reminder_close", re.compile(r"</system-reminder>", re.I)),
        ("whenever_you_read_a_file", re.compile(r"Whenever you read a file", re.I)),
    ]


def string_values_for_validation(value: Any, *, skip_keys: set[str]) -> list[str]:
    """Collect string values only, not JSON object keys, for leak validation."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            out.extend(string_values_for_validation(item, skip_keys=skip_keys))
    elif isinstance(value, dict):
        for key, item in value.items():
            if str(key) in skip_keys:
                continue
            out.extend(string_values_for_validation(item, skip_keys=skip_keys))
    return out


def key_names_for_validation(value: Any, *, skip_keys: set[str]) -> list[str]:
    out: list[str] = []
    if isinstance(value, list):
        for item in value:
            out.extend(key_names_for_validation(item, skip_keys=skip_keys))
    elif isinstance(value, dict):
        for key, item in value.items():
            key_s = str(key)
            if key_s in skip_keys:
                continue
            out.append(f"__KEY__:{key_s}")
            out.extend(key_names_for_validation(item, skip_keys=skip_keys))
    return out


def validate_output_file(path: Path, root: Path | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        pieces: list[str] = []
        skip_keys = {"renderer", "quality", "strip_metadata"}
        for _lineno, obj in iter_jsonl(path):
            pieces.extend(string_values_for_validation(obj, skip_keys=skip_keys))
            pieces.extend(key_names_for_validation(obj, skip_keys=skip_keys))
        text = "\n".join(pieces)
    elif path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        text = "\n".join(string_values_for_validation(obj, skip_keys={"validation_forbidden_pattern_hits"}))
    else:
        text = path.read_text(encoding="utf-8")
    rows = []
    for name, pat in forbidden_patterns():
        if pat.search(text):
            file_label = str(path.relative_to(root)) if root is not None else str(path)
            rows.append({"file": file_label, "pattern": name})
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def config_for_hash(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "include_final_message", "include_todowrite", "keep_reasoning_content", "drop_action_descriptions",
        "normalize_user_prompt", "keep_additional_user_messages", "drop_scratch_reads",
        "drop_setup_commands", "drop_destructive_commands", "drop_build_install_commands",
        "max_raw_tool_result_chars", "max_adp_observation_chars", "max_adp_action_chars", "memory_template", "edit_body_mode",
    ]
    return {k: getattr(args, k) for k in keys}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministically strip harness/environment junk from ADP, raw Claude JSONL, and memory JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --adp claude_code_adp.jsonl --out-dir stripped
  %(prog)s --adp claude_code_adp.jsonl --raw fd5f1dd2.jsonl c86d3ad2.jsonl --out-dir stripped
  %(prog)s --adp claude_code_adp.jsonl --memory-jsonl memories.jsonl --out-dir stripped
  %(prog)s --adp claude_code_adp.jsonl --memory-converter ./adp_to_memory.py --out-dir stripped --memory-template openhands
        """.strip(),
    )
    p.add_argument("--adp", type=Path, required=True, help="Input ADP JSONL file.")
    p.add_argument("--raw", nargs="*", type=Path, default=[], help="Optional raw Claude Code JSONL files to render as stripped transcripts.")
    p.add_argument("--memory-jsonl", type=Path, default=None, help="Optional existing memory JSONL to strip by manifest indices.")
    p.add_argument("--memory-converter", type=Path, default=None, help="Optional adp_to_memory.py path. If provided, runs it on stripped ADP.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--memory-template", choices=["generic", "swe-agent", "openhands"], default="openhands")

    # Conservative defaults.
    p.add_argument("--include-final-message", action="store_true", help="Keep final assistant text/message_action. Default: drop.")
    p.add_argument("--drop-action-descriptions", action=argparse.BooleanOptionalAction, default=True, help="Remove Claude tool/action descriptions from kept ADP actions. Default: true.")
    p.add_argument("--include-todowrite", action="store_true", help="Keep TodoWrite actions. Default: drop.")
    p.add_argument("--keep-reasoning-content", action="store_true", help="Keep Claude thinking/reasoning_content. Default: drop.")
    p.add_argument("--normalize-user-prompt", action=argparse.BooleanOptionalAction, default=True, help="Replace first user prompt with normalized prior issue metadata. Default: true.")
    p.add_argument("--keep-additional-user-messages", action="store_true", help="Keep user messages after the first prompt. Default: drop.")
    p.add_argument("--drop-scratch-reads", action=argparse.BooleanOptionalAction, default=True, help="Drop reads/greps of scratch/debug files. Default: true.")
    p.add_argument("--drop-setup-commands", action=argparse.BooleanOptionalAction, default=True, help="Drop setup commands. Default: true.")
    p.add_argument("--drop-destructive-commands", action=argparse.BooleanOptionalAction, default=True, help="Drop destructive commands. Default: true.")
    p.add_argument("--drop-build-install-commands", action=argparse.BooleanOptionalAction, default=True, help="Drop build/install commands. Default: true.")
    p.add_argument("--max-raw-tool-result-chars", type=int, default=2000, help="Cap raw tool result excerpts in stripped raw transcript.")
    p.add_argument("--max-adp-observation-chars", type=int, default=2000, help="Cap observation excerpts in rendered stripped-ADP injection text.")
    p.add_argument("--max-adp-action-chars", type=int, default=2000, help="Cap action payloads in rendered stripped-ADP injection text.")
    p.add_argument("--edit-body-mode", choices=["hashes-only", "excerpts"], default="hashes-only", help="Render edit bodies as hashes only or include excerpts. Default: hashes-only.")
    p.add_argument("--stats", action="store_true", help="Print stats to stderr.")
    p.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True, help="Scan outputs for forbidden strings. Default: true.")
    return p.parse_args(argv)


def _source_match_key(value: str) -> str:
    name = Path(normalize_text(value)).name
    stem = Path(name).stem
    # Claude examples often get copied as c3091b8c(1).jsonl, c3091b8c(5).jsonl.
    return re.sub(r"\([^)]*\)$", "", stem)


def match_manifest_for_raw(raw_path: Path, manifests: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw_key = _source_match_key(raw_path.name)
    candidates: list[dict[str, Any]] = []
    for manifest in manifests:
        source_file = normalize_text(manifest.get("source_file") or "")
        source_key = _source_match_key(source_file) if source_file else ""
        traj_key = _source_match_key(normalize_text(manifest.get("trajectory_id") or ""))
        if raw_key and raw_key in {source_key, traj_key}:
            candidates.append(manifest)
    if len(candidates) == 1:
        return candidates[0]
    if len(manifests) == 1:
        return manifests[0]
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.adp.exists():
        print(f"ERROR: ADP input not found: {args.adp}", file=sys.stderr)
        return 2
    for raw in args.raw:
        if not raw.exists():
            print(f"ERROR: raw input not found: {raw}", file=sys.stderr)
            return 2
    if args.memory_jsonl and not args.memory_jsonl.exists():
        print(f"ERROR: memory JSONL input not found: {args.memory_jsonl}", file=sys.stderr)
        return 2
    if args.memory_converter and not args.memory_converter.exists():
        print(f"ERROR: memory converter not found: {args.memory_converter}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_dir = args.out_dir / "raw_transcripts"
    mem_text_dir = args.out_dir / "memory_text"

    strip_config_hash = sha256_json({"strip_policy_version": STRIP_POLICY_VERSION, "config": config_for_hash(args)})

    manifests: list[dict[str, Any]] = []
    stripped_trajs: list[dict[str, Any]] = []
    for record_index, (_lineno, traj) in enumerate(iter_jsonl(args.adp), start=1):
        manifest, stripped = build_manifest_and_stripped_traj(traj, args, record_index)
        manifest["strip_config_hash"] = strip_config_hash
        stripped["details"]["strip_metadata"]["strip_config_hash"] = strip_config_hash
        # Recompute now that config hash has been filled.
        manifest["stripped_trajectory_sha256"] = sha256_json(stripped)
        manifests.append(manifest)
        stripped_trajs.append(stripped)
        if args.stats:
            print(
                f"{manifest['trajectory_id']}: kept={manifest['decision_counts'].get('keep', 0)} "
                f"dropped={manifest['decision_counts'].get('drop', 0)}",
                file=sys.stderr,
            )

    stripped_adp_audit_path = args.out_dir / "stripped_adp.audit.jsonl"
    stripped_adp_injection_path = args.out_dir / "stripped_adp_injection.jsonl"
    manifest_path = args.out_dir / "strip_manifest.jsonl"
    write_jsonl(stripped_adp_audit_path, stripped_trajs)
    write_jsonl(stripped_adp_injection_path, [make_injection_safe_adp_traj(traj, args) for traj in stripped_trajs])
    write_jsonl(manifest_path, manifests)
    rendered_adp_paths = write_rendered_adp_injections(stripped_trajs, args.out_dir, args)

    raw_manifests: list[dict[str, Any]] = []
    if args.raw:
        raw_out_dir.mkdir(parents=True, exist_ok=True)
        for idx, raw_path in enumerate(args.raw, start=1):
            matched_manifest = match_manifest_for_raw(raw_path, manifests)
            transcript, raw_manifest = render_raw_transcript(raw_path, args, matched_manifest)
            if matched_manifest is None:
                raw_manifest.setdefault("warnings", []).append("No ADP manifest matched this raw file; used policy fallback.")
            out_path = raw_out_dir / f"{idx:06d}_{raw_path.stem}.txt"
            out_path.write_text(transcript, encoding="utf-8", newline="\n")
            raw_manifest["output_text_file"] = str(out_path.relative_to(args.out_dir))
            raw_manifests.append(raw_manifest)
        write_jsonl(args.out_dir / "raw_strip_manifest.jsonl", raw_manifests)

    if args.memory_jsonl:
        by_id = manifest_by_id(manifests)
        stripped_memory: list[dict[str, Any]] = []
        mem_text_dir.mkdir(parents=True, exist_ok=True)
        for idx, (_lineno, memory) in enumerate(iter_jsonl(args.memory_jsonl), start=1):
            tid = normalize_text(memory.get("source_trajectory_id") or memory.get("trajectory_id") or "")
            manifest = by_id.get(tid)
            if manifest is None:
                # Deterministic error-ish record rather than guessing.
                out = deepcopy(memory)
                out.setdefault("strip_metadata", {})
                out["strip_metadata"] = {
                    "strip_policy_version": STRIP_POLICY_VERSION,
                    "error": f"No manifest found for source_trajectory_id={tid!r}",
                }
            else:
                out = strip_memory_record(memory, manifest, args)
            stripped_memory.append(out)
            text = normalize_text(out.get("rendered_text") or "")
            name = tid or f"memory-{idx}"
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")[:120] or f"memory-{idx}"
            (mem_text_dir / f"{idx:06d}_{safe}.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
        write_jsonl(args.out_dir / "stripped_memory.jsonl", stripped_memory)

    if args.memory_converter:
        generated_jsonl = args.out_dir / "generated_memory_from_stripped_adp.jsonl"
        generated_text_dir = args.out_dir / "generated_memory_text_from_stripped_adp"
        cmd = [
            sys.executable,
            str(args.memory_converter),
            str(stripped_adp_audit_path),
            "--output-jsonl",
            str(generated_jsonl),
            "--output-text-dir",
            str(generated_text_dir),
            "--template",
            args.memory_template,
            "--no-keep-excluded-edit-details",
            "--edit-body-mode",
            args.edit_body_mode,
        ]
        if args.include_todowrite:
            cmd.append("--include-todowrite")
        if args.keep_reasoning_content:
            cmd.append("--include-reasoning")
        subprocess.run(cmd, check=True)

    report = {
        "script": SCRIPT_NAME,
        "strip_policy_version": STRIP_POLICY_VERSION,
        "strip_config_hash": strip_config_hash,
        "adp_input": args.adp.name,
        "adp_input_sha256": sha256_text(args.adp.read_text(encoding="utf-8")),
        "outputs": {
            "stripped_adp_audit_jsonl": "stripped_adp.audit.jsonl",
            "stripped_adp_injection_jsonl": "stripped_adp_injection.jsonl",
            "strip_manifest_jsonl": "strip_manifest.jsonl",
            "stripped_adp_rendered_dir": "stripped_adp_rendered",
            "raw_transcripts_dir": "raw_transcripts" if args.raw else "",
            "raw_strip_manifest_jsonl": "raw_strip_manifest.jsonl" if args.raw else "",
            "stripped_memory_jsonl": "stripped_memory.jsonl" if args.memory_jsonl else "",
            "generated_memory_jsonl": "generated_memory_from_stripped_adp.jsonl" if args.memory_converter else "",
            "generated_memory_text_dir": "generated_memory_text_from_stripped_adp" if args.memory_converter else "",
        },
        "trajectory_count": len(stripped_trajs),
        "aggregate_decision_counts": dict(sorted(sum((Counter(m["decision_counts"]) for m in manifests), Counter()).items())),
        "aggregate_reason_counts": dict(sorted(sum((Counter(m["reason_counts"]) for m in manifests), Counter()).items())),
        "raw_file_count": len(args.raw),
        "warnings": [],
    }

    validation_hits: list[dict[str, Any]] = []
    if args.validate:
        candidate_files = [stripped_adp_injection_path]
        candidate_files.extend(rendered_adp_paths)
        if args.raw:
            candidate_files.extend(sorted(raw_out_dir.glob("*.txt")))
        if args.memory_converter:
            candidate_files.append(args.out_dir / "generated_memory_from_stripped_adp.jsonl")
            candidate_files.extend(sorted((args.out_dir / "generated_memory_text_from_stripped_adp").glob("*.txt")))
        for path in candidate_files:
            validation_hits.extend(validate_output_file(path, root=args.out_dir))
        report["validation_forbidden_pattern_hits"] = validation_hits
        if validation_hits:
            report["warnings"].append("Forbidden-pattern validation found possible leaks. Inspect validation_forbidden_pattern_hits.")

    (args.out_dir / "strip_report.json").write_text(stable_json_dumps(report, pretty=True) + "\n", encoding="utf-8", newline="\n")

    if args.stats:
        print(f"wrote {stripped_adp_audit_path}", file=sys.stderr)
        print(f"wrote {stripped_adp_injection_path}", file=sys.stderr)
        print(f"wrote {manifest_path}", file=sys.stderr)
        print(f"wrote {args.out_dir / 'strip_report.json'}", file=sys.stderr)
        if validation_hits:
            print(f"WARNING: validation hits: {len(validation_hits)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
