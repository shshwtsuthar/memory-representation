#!/usr/bin/env python3
"""Mechanism attribution analysis for ContextBench prior-context oracle wins.

The script reads the final 95-target posthoc inputs in this analysis directory
and a local mirror of the final OpenHands run artifacts. If the mirror is
missing, pass --fetch-remote with SSHPASS set to fetch only read-only artifacts
from /mnt/data/shashwat.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REMOTE_CACHE = ROOT / "remote" / "execution_full_qwen36_65k_fix1_minimal"

CONDITIONS = ["no_memory", "raw", "adp", "memory"]
PRIOR_CONDITIONS = ["raw", "adp", "memory"]
EXCLUDED = "django__django-28147"
EXPECTED_VALID_TARGETS = 95
EXPECTED_ORACLE_PRIOR = 28
EXPECTED_ORACLE_ALL = 29

REMOTE_RUN_ROOT = (
    "/mnt/data/shashwat/openhands-adp-memory/data/contextbench_phase2/"
    "execution_full_qwen36_65k_fix1/runs"
)

SIDE_CAR_FILES = {
    "prompt.txt",
    "stderr.log",
    "run_meta.json",
    "patch.diff",
    "prediction.json",
}

PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"((?:/?(?:testbed|workspace|swebench_9_15|contextbench_run|mnt|home)[A-Za-z0-9_./-]*|"
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+)"
    r"\.(?:py|pyx|pxd|rst|md|txt|json|yaml|yml|cfg|ini|toml|diff|patch|csv))"
)
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
ANSI_RE = re.compile(r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|[=><]|\][^\x07]*(?:\x07|\x1b\\))")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

FAILURE_LINE_RE = re.compile(
    r"(Traceback|AssertionError|AttributeError|TypeError|ValueError|KeyError|"
    r"ImportError|RuntimeError|Exception|FAILED|FAIL:|ERROR:|Error:|"
    r"raises?:|No matches found)",
    re.IGNORECASE,
)

TEST_CMD_RE = re.compile(
    r"\b(pytest|tox|unittest|runtests\.py|manage\.py\s+test|"
    r"python[0-9.]*\s+[^;\n]*(?:test_|_test|tests/|repro|bug))\b",
    re.IGNORECASE,
)

SEARCH_CMD_RE = re.compile(r"(^|\s)(rg|grep|find|ack|ag)(\s|$)")
FILE_READ_CMD_RE = re.compile(r"(^|\s)(cat|sed|head|tail|nl|less|more)(\s|$)")
GIT_CMD_RE = re.compile(r"(^|\s)git(\s|$)")
ENV_CMD_RE = re.compile(
    r"(^|\s)(pip|python[0-9.]*\s+--version|python[0-9.]*\s+-V|"
    r"which|pwd|ls|conda|mamba|uname|env)(\s|$)",
    re.IGNORECASE,
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "when",
    "where",
    "then",
    "than",
    "have",
    "has",
    "are",
    "was",
    "were",
    "will",
    "would",
    "should",
    "could",
    "issue",
    "error",
    "test",
    "tests",
    "file",
    "files",
    "class",
    "function",
    "method",
    "return",
    "expected",
    "actual",
    "current",
    "prior",
    "context",
    "repository",
}


def cap(text: Any, n: int = 320) -> str:
    s = "" if text is None else str(text)
    s = ANSI_RE.sub("", s)
    s = CONTROL_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[: n - 3] + "..." if len(s) > n else s


def uniq(items: Iterable[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = normalize_path(item) if looks_like_path(item) else str(item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def join(items: Iterable[str], limit: int | None = 80) -> str:
    return "|".join(uniq(items, limit))


def split_cell(cell: Any) -> list[str]:
    if cell is None:
        return []
    text = str(cell).strip()
    if not text:
        return []
    parts = re.split(r"[;|]", text)
    return uniq(p.strip() for p in parts if p.strip())


def looks_like_path(s: str) -> bool:
    return "/" in s or "." in Path(str(s)).name


def normalize_path(path: Any) -> str:
    s = "" if path is None else str(path).strip().strip("\"'`,:;")
    s = ANSI_RE.sub("", s)
    s = s.replace("\\", "/")
    prefixes = [
        "/testbed/",
        "testbed/",
        "./testbed/",
        "/workspace/",
        "workspace/",
        "./workspace/",
        "./swebench_9_15/testbed/",
        "swebench_9_15/testbed/",
        "./swebench_9_15/",
        "swebench_9_15/",
        "a/",
        "b/",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if s.startswith(prefix) and len(s) > len(prefix):
                s = s[len(prefix) :]
                changed = True
    s = re.sub(r"^\./", "", s)
    s = re.sub(r"//+", "/", s)
    common_roots = (
        "astropy/",
        "django/",
        "lib/",
        "matplotlib/",
        "pylint/",
        "requests/",
        "sklearn/",
        "sphinx/",
        "sympy/",
        "tests/",
        "xarray/",
    )
    if len(s) > 2 and s[0] in {"n", "t"} and any(s[1:].startswith(root) for root in common_roots):
        s = s[1:]
    return s


def extract_paths(text: str) -> list[str]:
    clean = ANSI_RE.sub("", text or "")
    clean = CONTROL_RE.sub("", clean)
    return uniq(m.group(1) for m in PATH_RE.finditer(clean))


def path_dirs(paths: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        parent = str(Path(p).parent)
        if parent and parent != ".":
            out.add(parent)
    return out


def bool_int(x: Any) -> int:
    if isinstance(x, str):
        return 1 if x.strip().lower() in {"1", "true", "yes", "resolved"} else 0
    return 1 if bool(x) else 0


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x == "" or x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x: Any) -> float:
    try:
        if x == "" or x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(errors="replace"))


def fetch_remote_artifacts(refresh: bool = False) -> None:
    complete = REMOTE_CACHE / ".complete"
    if complete.exists() and not refresh:
        print(f"Using cached remote artifact mirror: {REMOTE_CACHE}")
        return
    if "SSHPASS" not in os.environ:
        raise SystemExit("Set SSHPASS before using --fetch-remote.")

    REMOTE_CACHE.mkdir(parents=True, exist_ok=True)
    remote_code = rf"""
import tarfile
from pathlib import Path

root = Path({REMOTE_RUN_ROOT!r})
sidecars = {sorted(SIDE_CAR_FILES)!r}

def add_file(tf, p):
    if not p.is_file():
        return
    rel = p.relative_to(root)
    tf.add(p, arcname=str(rel), recursive=False)

with tarfile.open(fileobj=__import__('sys').stdout.buffer, mode='w|') as tf:
    for run_dir in sorted(root.glob('*/*')):
        if not run_dir.is_dir():
            continue
        for name in sidecars:
            add_file(tf, run_dir / name)
        for p in sorted(run_dir.glob('.openhands_home/.openhands/conversations/*/base_state.json')):
            add_file(tf, p)
        for p in sorted(run_dir.glob('.openhands_home/.openhands/conversations/*/events/event-*.json')):
            add_file(tf, p)
"""
    cmd = [
        "sshpass",
        "-e",
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        "shashwat@10.151.160.71",
        "python3",
        "-",
    ]
    print(f"Fetching minimal artifact mirror into {REMOTE_CACHE}")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(remote_code.encode("utf-8"))
    proc.stdin.close()
    extracted = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for member in tf:
                rel = Path(member.name)
                if rel.is_absolute() or ".." in rel.parts:
                    raise RuntimeError(f"Unsafe tar member: {member.name}")
                tf.extract(member, REMOTE_CACHE)
                extracted += 1
    finally:
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"Remote fetch failed rc={rc}: {stderr}")
    complete.write_text(f"extracted_files={extracted}\n", encoding="utf-8")
    print(f"Fetched {extracted} files")


def validate_oracles(paired_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in paired_rows:
        iid = row["instance_id"]
        if iid == EXCLUDED:
            continue
        out = dict(row)
        for cond in CONDITIONS:
            out[f"resolved_{cond}"] = safe_int(row.get(f"resolved_{cond}", 0))
            out[f"empty_patch_{cond}"] = safe_int(row.get(f"empty_patch_{cond}", 0))
            out[f"patch_failed_{cond}"] = safe_int(row.get(f"patch_failed_{cond}", 0))
        out["oracle_prior"] = max(out[f"resolved_{c}"] for c in PRIOR_CONDITIONS)
        out["oracle_all"] = max(out[f"resolved_{c}"] for c in CONDITIONS)
        rows.append(out)
    n = len(rows)
    oracle_prior = sum(r["oracle_prior"] for r in rows)
    oracle_all = sum(r["oracle_all"] for r in rows)
    counts = {c: sum(r[f"resolved_{c}"] for r in rows) for c in CONDITIONS}
    if n != EXPECTED_VALID_TARGETS or oracle_prior != EXPECTED_ORACLE_PRIOR or oracle_all != EXPECTED_ORACLE_ALL:
        raise SystemExit(
            "Oracle count discrepancy: "
            f"n={n}, oracle_prior={oracle_prior}, oracle_all={oracle_all}, "
            f"condition_counts={counts}"
        )
    return rows, {"n": n, "oracle_prior": oracle_prior, "oracle_all": oracle_all, "counts": counts}


def parse_event_index(path: Path) -> int:
    m = re.search(r"event-(\d+)-", path.name)
    return int(m.group(1)) if m else 10**9


def event_files(run_dir: Path) -> list[Path]:
    paths = sorted(
        run_dir.glob(".openhands_home/.openhands/conversations/*/events/event-*.json"),
        key=parse_event_index,
    )
    return paths


def parse_events_from_files(run_dir: Path) -> list[dict[str, Any]]:
    events = []
    for p in event_files(run_dir):
        try:
            obj = load_json(p)
        except Exception:
            continue
        if isinstance(obj, dict):
            obj["_artifact_path"] = str(p)
            events.append(obj)
    return events


def parse_events_from_stdout(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "stdout.jsonl"
    if not path.exists():
        return []
    text = path.read_text(errors="replace")
    decoder = json.JSONDecoder()
    events = []
    for part in text.split("--JSON Event--")[1:]:
        start = part.find("{")
        if start < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(part[start:])
        except Exception:
            continue
        if isinstance(obj, dict):
            obj["_artifact_path"] = str(path)
            events.append(obj)
    return events


def event_observation_text(event: dict[str, Any]) -> str:
    obs = event.get("observation") or {}
    chunks: list[str] = []
    if isinstance(obs, dict):
        content = obs.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    chunks.append(str(item.get("text", "")))
                else:
                    chunks.append(str(item))
        elif isinstance(content, str):
            chunks.append(content)
        for key in ["error", "new_content", "old_content"]:
            val = obs.get(key)
            if isinstance(val, str):
                chunks.append(val)
    for key in ["error", "content"]:
        val = event.get(key)
        if isinstance(val, str):
            chunks.append(val)
    return "\n".join(c for c in chunks if c)


def parse_tool_args(event: dict[str, Any]) -> dict[str, Any]:
    action = event.get("action")
    if isinstance(action, dict):
        return dict(action)
    tool_call = event.get("tool_call")
    if isinstance(tool_call, dict):
        raw = tool_call.get("arguments")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {"raw_arguments": raw}
    return {}


def command_from_action(tool_name: str, args: dict[str, Any]) -> str:
    cmd = args.get("command")
    if tool_name == "terminal":
        return str(cmd or "")
    if tool_name == "file_editor":
        path = args.get("path") or ""
        return f"{cmd or ''} {path}".strip()
    if tool_name == "think":
        return cap(args.get("thought", ""), 180)
    if tool_name == "task_tracker":
        return str(cmd or args.get("command") or "")
    return str(cmd or args.get("raw_arguments") or "")


def normalize_command_type(tool_name: str, command: str, args: dict[str, Any]) -> str:
    c = (command or "").strip().lower()
    if tool_name == "file_editor":
        op = str(args.get("command") or "").lower()
        if op == "view":
            return "file_read"
        if op in {"create", "str_replace", "insert", "undo_edit", "edit"}:
            return "file_edit"
        return "diagnostic"
    if tool_name in {"think", "task_tracker"}:
        return "diagnostic"
    if tool_name == "finish":
        return "other"
    if GIT_CMD_RE.search(c):
        return "git"
    if TEST_CMD_RE.search(c):
        if re.search(r"\b(pytest|tox|unittest|runtests\.py|manage\.py\s+test)\b", c):
            return "test"
        return "repro"
    if SEARCH_CMD_RE.search(c):
        return "search"
    if FILE_READ_CMD_RE.search(c):
        return "file_read"
    if "apply_patch" in c or "git apply" in c or "cat >" in c:
        return "file_edit"
    if ENV_CMD_RE.search(c):
        return "environment"
    if "python" in c or "inspect" in c:
        return "diagnostic"
    return "other"


def shell_read_paths(command: str) -> list[str]:
    if not FILE_READ_CMD_RE.search(command.lower()):
        return []
    return extract_paths(command)


def shell_edit_paths(command: str) -> list[str]:
    c = command.lower()
    if not ("apply_patch" in c or "git apply" in c or "cat >" in c or "tee " in c):
        return []
    return extract_paths(command)


def failure_excerpt(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        if FAILURE_LINE_RE.search(line):
            lines.append(line.strip())
    return cap(" | ".join(lines), 320)


def build_tool_timeline(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    valid_ids = {r["instance_id"] for r in valid_rows}
    for iid in sorted(valid_ids):
        for cond in CONDITIONS:
            run_dir = REMOTE_CACHE / iid / cond
            events = parse_events_from_files(run_dir)
            if not events:
                events = parse_events_from_stdout(run_dir)
            obs_by_call: dict[str, dict[str, Any]] = {}
            for event in events:
                if event.get("kind") == "ObservationEvent":
                    call_id = event.get("tool_call_id")
                    if call_id:
                        obs_by_call[str(call_id)] = event
            step = 0
            for event in events:
                kind = event.get("kind")
                if kind not in {"ActionEvent", "AgentErrorEvent"}:
                    continue
                tool_name = str(event.get("tool_name") or "")
                call_id = str(event.get("tool_call_id") or "")
                args = parse_tool_args(event)
                command = command_from_action(tool_name, args)
                obs = obs_by_call.get(call_id, {})
                obs_text = event_observation_text(obs)
                event_text = "\n".join([command, json.dumps(args, sort_keys=True, default=str), obs_text])
                norm = normalize_command_type(tool_name, command, args)
                file_paths_mentioned = extract_paths(event_text)
                file_paths_read: list[str] = []
                file_paths_edited: list[str] = []
                if tool_name == "file_editor":
                    path = args.get("path")
                    op = str(args.get("command") or "").lower()
                    if path and op == "view":
                        file_paths_read.append(normalize_path(path))
                    elif path and op in {"create", "str_replace", "insert", "undo_edit", "edit"}:
                        file_paths_edited.append(normalize_path(path))
                elif tool_name == "terminal":
                    file_paths_read.extend(shell_read_paths(command))
                    file_paths_edited.extend(shell_edit_paths(command))
                tests_run = [command] if norm in {"test", "repro"} else []
                tool_error = kind == "AgentErrorEvent"
                obs_obj = obs.get("observation") if isinstance(obs, dict) else None
                if isinstance(obs_obj, dict):
                    tool_error = tool_error or bool(obs_obj.get("is_error"))
                step += 1
                timeline.append(
                    {
                        "instance_id": iid,
                        "condition": cond,
                        "step_index": step,
                        "event_type": kind,
                        "tool_name": tool_name,
                        "command": cap(command, 500),
                        "normalized_command_type": norm,
                        "file_paths_mentioned": join(file_paths_mentioned),
                        "file_paths_read": join(file_paths_read),
                        "file_paths_edited": join(file_paths_edited),
                        "tests_run": join(tests_run, 20),
                        "failure_signature_excerpt": failure_excerpt(obs_text),
                        "observation_excerpt_capped": cap(obs_text, 500),
                        "tool_error_bool": int(tool_error),
                        "artifact_path": str(event.get("_artifact_path") or run_dir),
                    }
                )
    return timeline


def parse_patch_files(diff_text: str) -> list[str]:
    files = []
    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*?) b/(.*)$", line)
            if m:
                candidate = m.group(2)
                if candidate != "/dev/null":
                    files.append(candidate)
    return uniq(files)


def current_issue_text(prompt_text: str) -> str:
    m = re.search(r"CURRENT_ISSUE\n(.*?)\nEND_CURRENT_ISSUE", prompt_text, re.S)
    return m.group(1) if m else ""


def prior_context_text(prompt_text: str) -> str:
    m = re.search(r"PRIOR_CONTEXT\n(.*?)\nEND_PRIOR_CONTEXT", prompt_text, re.S)
    return m.group(1) if m else ""


def symbols_from_text(text: str, limit: int = 120) -> list[str]:
    symbols: list[str] = []
    for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_.]*)`", text):
        symbols.append(m.group(1).split(".")[-1])
    for m in re.finditer(r"\b(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", text):
        symbols.append(m.group(1))
    for m in IDENT_RE.finditer(text):
        tok = m.group(0)
        if tok.lower() in STOPWORDS:
            continue
        if "_" in tok or re.search(r"[a-z][A-Z]|[A-Z][a-z]", tok):
            symbols.append(tok)
    return uniq(symbols, limit)


def keyword_tokens(text: str, limit: int = 80) -> list[str]:
    counts = Counter()
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text.lower()):
        if tok in STOPWORDS:
            continue
        counts[tok] += 1
    return [tok for tok, _ in counts.most_common(limit)]


def extract_test_commands(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if TEST_CMD_RE.search(stripped):
            stripped = re.sub(r"^\[[^\]]+\]\s*", "", stripped)
            stripped = re.sub(r"^command:\s*", "", stripped, flags=re.I)
            out.append(stripped)
    return uniq(out, 80)


def extract_search_anchors(text: str) -> list[str]:
    anchors: list[str] = []
    for m in re.finditer(r'"pattern"\s*:\s*"((?:\\.|[^"\\])*)"', text):
        try:
            anchors.append(bytes(m.group(1), "utf-8").decode("unicode_escape"))
        except Exception:
            anchors.append(m.group(1))
    for line in text.splitlines():
        if re.search(r"\b(rg|grep)\b", line):
            quoted = re.findall(r'"([^"\n]{2,120})"|\'([^\'\n]{2,120})\'', line)
            for a, b in quoted:
                anchors.append(a or b)
            m = re.search(r"\b(?:rg|grep)(?:\s+-[A-Za-z0-9]+)*\s+([^|;&]{2,120})", line)
            if m:
                anchors.append(m.group(1).strip())
    return uniq((cap(a, 120) for a in anchors), 80)


def extract_failure_lines(text: str) -> list[str]:
    return uniq((cap(line.strip(), 220) for line in text.splitlines() if FAILURE_LINE_RE.search(line)), 80)


def extract_observation_excerpts(text: str) -> list[str]:
    excerpts = []
    for i, line in enumerate(text.splitlines()):
        if "TOOL RESULT" in line or "Observation" in line or FAILURE_LINE_RE.search(line):
            excerpts.append(cap(line, 220))
            if i + 1 < len(text.splitlines()):
                pass
    return uniq(excerpts, 60)


def extract_patch_shape_hints(text: str) -> list[str]:
    hints = []
    for line in text.splitlines():
        low = line.lower()
        if (
            line.startswith("diff --git")
            or line.startswith("@@")
            or "str_replace" in low
            or "old_str" in low
            or "new_str" in low
            or "return " in low
            or "elif " in low
            or "if " in low and ("none" in low or "not" in low)
        ):
            hints.append(cap(line, 220))
    return uniq(hints, 80)


def classify_prior_file_roles(text: str) -> tuple[list[str], list[str]]:
    inspected: list[str] = []
    edited: list[str] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        context = "\n".join(lines[max(0, idx - 2) : min(len(lines), idx + 3)])
        paths = extract_paths(line)
        if not paths:
            continue
        low = context.lower()
        if any(k in low for k in ["read", "view", "grep", "glob", "rg", "cat", "sed", "inspected", "opened"]):
            inspected.extend(paths)
        if any(k in low for k in ["edit", "write", "create", "patch", "diff --git", "modified", "str_replace"]):
            edited.extend(paths)
    return uniq(inspected, 120), uniq(edited, 120)


def build_prompt_indices(
    valid_rows: list[dict[str, Any]],
    overlap_by_id: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    target_info: dict[str, dict[str, Any]] = {}
    for row in valid_rows:
        iid = row["instance_id"]
        prompt_path = REMOTE_CACHE / iid / "raw" / "prompt.txt"
        prompt_text = prompt_path.read_text(errors="replace") if prompt_path.exists() else ""
        issue = current_issue_text(prompt_text)
        target_info[iid] = {
            "issue_text": issue,
            "keywords": keyword_tokens(issue),
            "symbols": symbols_from_text(issue),
            "failure_lines": extract_failure_lines(issue),
        }
    for row in valid_rows:
        iid = row["instance_id"]
        overlap = overlap_by_id.get(iid, {})
        gold_files = split_cell(overlap.get("target_gold_files", ""))
        gold_basenames = [Path(p).name for p in gold_files]
        gold_dirs = list(path_dirs(gold_files))
        target_keywords = target_info[iid]["keywords"]
        for cond in PRIOR_CONDITIONS:
            prompt_path = REMOTE_CACHE / iid / cond / "prompt.txt"
            prompt_text = prompt_path.read_text(errors="replace") if prompt_path.exists() else ""
            prior = prior_context_text(prompt_text)
            files_mentioned = extract_paths(prior)
            inspected, edited = classify_prior_file_roles(prior)
            test_files = [p for p in files_mentioned if "/tests/" in p or Path(p).name.startswith("test_") or "_test" in Path(p).name]
            test_commands = extract_test_commands(prior)
            anchors = extract_search_anchors(prior)
            failures = extract_failure_lines(prior)
            symbols = symbols_from_text(prior)
            command_hints = []
            for line in prior.splitlines():
                if re.search(r"\b(TOOL|Bash|Read|Grep|Glob|Edit|Write|pytest|python|rg|grep|find)\b", line):
                    command_hints.append(cap(line, 180))
            patch_hints = extract_patch_shape_hints(prior)
            observation_excerpts = extract_observation_excerpts(prior)
            basename_mentions = sum(prior.count(name) for name in gold_basenames if name)
            dir_mentions = sum(prior.count(d) for d in gold_dirs if d)
            keyword_overlap = sum(1 for tok in target_keywords if re.search(rf"\b{re.escape(tok)}\b", prior.lower()))
            inv = {
                "instance_id": iid,
                "condition": cond,
                "prompt_path": str(prompt_path),
                "prior_context_chars": len(prior),
                "prior_files_mentioned": join(files_mentioned),
                "prior_files_mentioned_count": len(files_mentioned),
                "prior_files_inspected": join(inspected),
                "prior_files_inspected_count": len(inspected),
                "prior_files_edited": join(edited),
                "prior_files_edited_count": len(edited),
                "prior_test_files_mentioned": join(test_files),
                "prior_test_commands": join(test_commands),
                "grep_search_anchors": join(anchors),
                "error_failure_assertion_lines": join(failures),
                "function_class_symbol_names": join(symbols),
                "command_sequence_hints": join(command_hints, 40),
                "observation_excerpts": join(observation_excerpts, 40),
                "patch_shape_hints": join(patch_hints, 40),
                "target_gold_file_basename_mention_count": basename_mentions,
                "target_gold_directory_mention_count": dir_mentions,
                "target_issue_keyword_overlap_count": keyword_overlap,
            }
            inventory.append(inv)
            by_key[(iid, cond)] = inv
    return inventory, by_key, target_info


def token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{3,}", (text or "").lower()) if t not in STOPWORDS}


def similarity(a: str, b: str) -> float:
    aa = token_set(a)
    bb = token_set(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def build_evidence_overlap(
    inventory: list[dict[str, Any]],
    overlap_by_id: dict[str, dict[str, str]],
    target_info: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows = []
    by_key = {}
    for inv in inventory:
        iid = inv["instance_id"]
        cond = inv["condition"]
        overlap = overlap_by_id.get(iid, {})
        gold_files = set(split_cell(overlap.get("target_gold_files", "")))
        gold_dirs = path_dirs(gold_files)
        mentioned = set(split_cell(inv.get("prior_files_mentioned", "")))
        inspected = set(split_cell(inv.get("prior_files_inspected", "")))
        edited = set(split_cell(inv.get("prior_files_edited", "")))
        symbols = set(split_cell(inv.get("function_class_symbol_names", "")))
        target_symbols = set(target_info.get(iid, {}).get("symbols", []))
        prior_failures = split_cell(inv.get("error_failure_assertion_lines", ""))
        target_failures = target_info.get(iid, {}).get("failure_lines", [])
        failure_match = any(similarity(a, b) >= 0.18 or (a and a in b) or (b and b in a) for a in prior_failures for b in target_failures)
        prior_tests = split_cell(inv.get("prior_test_commands", ""))
        test_relevant = False
        for cmd in prior_tests:
            paths = extract_paths(cmd)
            if set(paths) & gold_files or path_dirs(paths) & gold_dirs:
                test_relevant = True
            if any(sym.lower() in cmd.lower() for sym in target_symbols):
                test_relevant = True
        mentions_file = bool(mentioned & gold_files) or any(Path(g).name in inv.get("prior_files_mentioned", "") for g in gold_files)
        mentions_dir = bool(path_dirs(mentioned) & gold_dirs) or any(d and d in inv.get("prior_files_mentioned", "") for d in gold_dirs)
        inspected_file = bool(inspected & gold_files)
        edited_file = bool(edited & gold_files)
        score = 0.0
        score += 3.0 if mentions_file else 0.0
        score += 1.5 if mentions_dir else 0.0
        score += 1.5 if inspected_file else 0.0
        score += 2.0 if edited_file else 0.0
        score += 1.0 if bool(symbols & target_symbols) else 0.0
        score += 2.0 if failure_match else 0.0
        score += 1.5 if test_relevant else 0.0
        score += min(2.0, safe_int(inv.get("target_gold_file_basename_mention_count")) * 0.2)
        score += min(1.0, safe_int(inv.get("target_gold_directory_mention_count")) * 0.1)
        score += min(2.0, safe_int(inv.get("target_issue_keyword_overlap_count")) * 0.1)
        row = {
            "instance_id": iid,
            "condition": cond,
            "evidence_mentions_target_gold_file_bool": int(mentions_file),
            "evidence_mentions_target_gold_dir_bool": int(mentions_dir),
            "evidence_mentions_target_gold_symbol_bool": int(bool(symbols & target_symbols)),
            "evidence_mentions_target_failure_signature_bool": int(failure_match),
            "evidence_contains_relevant_test_command_bool": int(test_relevant),
            "evidence_contains_prior_edited_target_gold_file_bool": int(edited_file),
            "evidence_contains_prior_inspected_target_gold_file_bool": int(inspected_file),
            "evidence_overlap_score": round(score, 3),
            "target_gold_files": join(gold_files),
            "target_gold_dirs": join(gold_dirs),
            "overlapping_symbols": join(sorted(symbols & target_symbols)),
            "failure_signature_overlap_excerpt": cap(" || ".join(prior_failures[:2] + target_failures[:2]), 500),
        }
        rows.append(row)
        by_key[(iid, cond)] = row
    return rows, by_key


def build_behavior_features(
    valid_rows: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    overlap_by_id: dict[str, dict[str, str]],
    inventory_by_key: dict[tuple[str, str], dict[str, Any]],
    target_info: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paired_by_id = {r["instance_id"]: r for r in valid_rows}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ev in timeline:
        grouped[(ev["instance_id"], ev["condition"])].append(ev)
    feature_rows: list[dict[str, Any]] = []
    feature_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    test_reuse_rows: list[dict[str, Any]] = []
    failure_reuse_rows: list[dict[str, Any]] = []
    for iid, paired in paired_by_id.items():
        overlap = overlap_by_id.get(iid, {})
        gold_files = set(split_cell(overlap.get("target_gold_files", "")))
        gold_dirs = path_dirs(gold_files)
        prior_inspected = set(split_cell(overlap.get("prior_trajectory_inspected_files", "")))
        prior_edited = set(split_cell(overlap.get("prior_trajectory_edited_files", "")))
        for cond in CONDITIONS:
            evs = grouped.get((iid, cond), [])
            evs = sorted(evs, key=lambda e: safe_int(e.get("step_index")))
            counts = Counter(e.get("normalized_command_type", "other") for e in evs if e.get("event_type") == "ActionEvent")
            read_events = [(safe_int(e["step_index"]), split_cell(e.get("file_paths_read", ""))) for e in evs]
            edit_events = [(safe_int(e["step_index"]), split_cell(e.get("file_paths_edited", ""))) for e in evs]
            all_read = [p for _, paths in read_events for p in paths]
            all_edited = [p for _, paths in edit_events for p in paths]
            all_commands = [e.get("command", "") for e in evs if e.get("command")]
            test_commands = [e.get("command", "") for e in evs if e.get("normalized_command_type") in {"test", "repro"}]
            failure_excerpts = [e.get("failure_signature_excerpt", "") for e in evs if e.get("failure_signature_excerpt")]
            first_gold_file_read = min(
                [step for step, paths in read_events if set(paths) & gold_files] or [None],
                key=lambda x: 10**9 if x is None else x,
            )
            first_gold_dir_read = min(
                [step for step, paths in read_events if path_dirs(paths) & gold_dirs] or [None],
                key=lambda x: 10**9 if x is None else x,
            )
            first_gold_file_edit = min(
                [step for step, paths in edit_events if set(paths) & gold_files] or [None],
                key=lambda x: 10**9 if x is None else x,
            )
            patch_path = REMOTE_CACHE / iid / cond / "patch.diff"
            patch_text = patch_path.read_text(errors="replace") if patch_path.exists() else ""
            patch_files = parse_patch_files(patch_text)
            patch_file_set = set(patch_files)
            inv = inventory_by_key.get((iid, cond), {})
            prior_test_cmds = split_cell(inv.get("prior_test_commands", "")) if cond in PRIOR_CONDITIONS else []
            prior_failures = split_cell(inv.get("error_failure_assertion_lines", "")) if cond in PRIOR_CONDITIONS else []
            ran_prior_test = any(command_reuses_prior(cmd, prior_test_cmds) for cmd in test_commands)
            reused_failure = any(similarity(run, prior) >= 0.18 or text_contains_key_phrase(run, prior) for run in failure_excerpts for prior in prior_failures)
            target_failures = target_info.get(iid, {}).get("failure_lines", [])
            reproduced_failure = reused_failure or any(similarity(run, target) >= 0.18 for run in failure_excerpts for target in target_failures)
            row = {
                "instance_id": iid,
                "condition": cond,
                "num_tool_calls": sum(1 for e in evs if e.get("event_type") == "ActionEvent"),
                "num_bash_commands": sum(1 for e in evs if e.get("tool_name") == "terminal" and e.get("event_type") == "ActionEvent"),
                "num_search_commands": counts["search"],
                "num_file_read_commands": counts["file_read"],
                "num_edit_commands": counts["file_edit"],
                "num_test_commands": counts["test"],
                "num_repro_commands": counts["repro"],
                "first_file_read": all_read[0] if all_read else "",
                "first_file_edited": all_edited[0] if all_edited else "",
                "first_gold_file_read_step": "" if first_gold_file_read is None else first_gold_file_read,
                "first_gold_dir_read_step": "" if first_gold_dir_read is None else first_gold_dir_read,
                "first_gold_file_edit_step": "" if first_gold_file_edit is None else first_gold_file_edit,
                "edited_target_gold_file_bool": int(bool(set(all_edited) & gold_files)),
                "edited_target_gold_dir_bool": int(bool(path_dirs(all_edited) & gold_dirs)),
                "ran_test_bool": int(bool(test_commands)),
                "ran_prior_test_command_bool": int(ran_prior_test),
                "reproduced_failure_bool": int(reproduced_failure),
                "patch_touches_gold_file_bool": int(bool(patch_file_set & gold_files)),
                "patch_touches_gold_dir_bool": int(bool(path_dirs(patch_files) & gold_dirs)),
                "patch_files": join(patch_files),
                "resolved": paired[f"resolved_{cond}"],
                "empty_patch": paired[f"empty_patch_{cond}"],
                "patch_failed": paired.get(f"patch_failed_{cond}", 0),
                "all_files_read": join(all_read),
                "all_files_edited": join(all_edited),
                "all_commands_run": join(all_commands, 120),
                "test_commands_run": join(test_commands, 80),
                "failure_signature_excerpts": join(failure_excerpts, 40),
                "touched_prior_inspected_file_bool": int(bool((set(all_read) | set(all_edited) | patch_file_set) & prior_inspected)),
                "touched_prior_edited_file_bool": int(bool((set(all_read) | set(all_edited) | patch_file_set) & prior_edited)),
                "tool_error_count": sum(safe_int(e.get("tool_error_bool")) for e in evs),
            }
            feature_rows.append(row)
            feature_by_key[(iid, cond)] = row
            test_reuse_rows.append(
                {
                    "instance_id": iid,
                    "condition": cond,
                    "prior_test_commands": join(prior_test_cmds),
                    "run_test_commands": join(test_commands),
                    "ran_prior_test_command_bool": int(ran_prior_test),
                    "resolved": paired[f"resolved_{cond}"],
                    "supporting_run_artifact": str(REMOTE_CACHE / iid / cond),
                }
            )
            failure_reuse_rows.append(
                {
                    "instance_id": iid,
                    "condition": cond,
                    "prior_failure_signature_excerpt": join(prior_failures, 10),
                    "run_failure_signature_excerpt": join(failure_excerpts, 10),
                    "target_failure_signature_excerpt": join(target_failures, 10),
                    "reused_prior_failure_signature_bool": int(reused_failure),
                    "reproduced_target_failure_bool": int(reproduced_failure),
                    "resolved": paired[f"resolved_{cond}"],
                    "supporting_run_artifact": str(REMOTE_CACHE / iid / cond),
                }
            )
    return feature_rows, feature_by_key, test_reuse_rows, failure_reuse_rows


def command_reuses_prior(command: str, prior_commands: list[str]) -> bool:
    c = re.sub(r"\s+", " ", command.lower()).strip()
    if not c:
        return False
    c_paths = set(extract_paths(command))
    for prior in prior_commands:
        p = re.sub(r"\s+", " ", prior.lower()).strip()
        if not p:
            continue
        if p in c or c in p:
            return True
        if c_paths and c_paths & set(extract_paths(prior)):
            return True
        if similarity(c, p) >= 0.35:
            return True
    return False


def text_contains_key_phrase(a: str, b: str) -> bool:
    aa = (a or "").lower()
    bb = (b or "").lower()
    for phrase in re.findall(r"[A-Za-z]+(?:Error|Exception)|'[^']{4,80}'|\"[^\"]{4,80}\"", a + "\n" + b):
        p = phrase.lower().strip("\"'")
        if len(p) >= 6 and p in aa and p in bb:
            return True
    return False


def case_types_for(row: dict[str, Any]) -> list[str]:
    raw = row["resolved_raw"]
    adp = row["resolved_adp"]
    mem = row["resolved_memory"]
    no = row["resolved_no_memory"]
    prior_sum = raw + adp + mem
    types = []
    if raw and not mem:
        types.append("raw_solved_memory_failed")
    if mem and not raw:
        types.append("memory_solved_raw_failed")
    if raw and not adp:
        types.append("raw_solved_adp_failed")
    if adp and not raw:
        types.append("adp_solved_raw_failed")
    if raw and adp and mem and not no:
        types.append("all_prior_solved_no_memory_failed")
    if no and prior_sum == 0:
        types.append("no_memory_solved_all_prior_failed")
    if prior_sum == 1:
        types.append("exactly_one_prior_representation_solved")
    if prior_sum == 2:
        types.append("at_least_two_prior_representations_solved_but_not_all")
    return types


def compare_steps(a: Any, b: Any) -> str:
    aa = safe_int(a, 10**9)
    bb = safe_int(b, 10**9)
    if aa == 10**9 and bb == 10**9:
        return "neither"
    if aa < bb:
        return "winner_earlier"
    if bb < aa:
        return "loser_earlier"
    return "tie"


def build_disagreement_cases(
    valid_rows: list[dict[str, Any]],
    evidence_by_key: dict[tuple[str, str], dict[str, Any]],
    features_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evidence_fields = [
        "evidence_mentions_target_gold_file_bool",
        "evidence_mentions_target_gold_dir_bool",
        "evidence_mentions_target_gold_symbol_bool",
        "evidence_mentions_target_failure_signature_bool",
        "evidence_contains_relevant_test_command_bool",
        "evidence_contains_prior_edited_target_gold_file_bool",
        "evidence_contains_prior_inspected_target_gold_file_bool",
    ]
    for row in valid_rows:
        cts = case_types_for(row)
        if not cts:
            continue
        iid = row["instance_id"]
        winners = [c for c in CONDITIONS if row[f"resolved_{c}"]]
        losers = [c for c in CONDITIONS if not row[f"resolved_{c}"]]
        prior_winners = [c for c in PRIOR_CONDITIONS if row[f"resolved_{c}"]]
        prior_losers = [c for c in PRIOR_CONDITIONS if not row[f"resolved_{c}"]]
        evidence_present = []
        for field in evidence_fields:
            if any(safe_int(evidence_by_key.get((iid, c), {}).get(field)) for c in prior_winners) and not all(
                safe_int(evidence_by_key.get((iid, c), {}).get(field)) for c in prior_losers
            ):
                evidence_present.append(field.replace("evidence_", "").replace("_bool", ""))
        winner_steps = [safe_int(features_by_key.get((iid, c), {}).get("first_gold_file_read_step"), 10**9) for c in winners]
        loser_steps = [safe_int(features_by_key.get((iid, c), {}).get("first_gold_file_read_step"), 10**9) for c in losers]
        winner_min = min(winner_steps) if winner_steps else 10**9
        loser_min = min(loser_steps) if loser_steps else 10**9
        winner_edited = any(safe_int(features_by_key.get((iid, c), {}).get("edited_target_gold_file_bool")) for c in winners)
        loser_edited = any(safe_int(features_by_key.get((iid, c), {}).get("edited_target_gold_file_bool")) for c in losers)
        winner_test_reuse = any(safe_int(features_by_key.get((iid, c), {}).get("ran_prior_test_command_bool")) for c in prior_winners)
        winner_failure_reuse = any(safe_int(features_by_key.get((iid, c), {}).get("reproduced_failure_bool")) for c in prior_winners)
        loser_empty_patch = any(safe_int(features_by_key.get((iid, c), {}).get("empty_patch")) for c in losers)
        loser_failed = any(
            safe_int(features_by_key.get((iid, c), {}).get("patch_failed"))
            or "failed" in str(features_by_key.get((iid, c), {}).get("failure_signature_excerpts", "")).lower()
            for c in losers
        )
        loser_irrelevant = any(
            features_by_key.get((iid, c), {}).get("first_file_read")
            and not safe_int(features_by_key.get((iid, c), {}).get("patch_touches_gold_dir_bool"))
            and not safe_int(features_by_key.get((iid, c), {}).get("patch_touches_gold_file_bool"))
            for c in losers
        )
        rows.append(
            {
                "instance_id": iid,
                "case_types": ";".join(cts),
                "condition_outcomes": ",".join(f"{c}:{row[f'resolved_{c}']}" for c in CONDITIONS),
                "winning_conditions": ";".join(winners),
                "losing_conditions": ";".join(losers),
                "evidence_present_in_winner_missing_from_loser": join(evidence_present),
                "winner_opened_gold_file_earlier": int(winner_min < loser_min),
                "winner_first_gold_file_read_step": "" if winner_min == 10**9 else winner_min,
                "loser_first_gold_file_read_step": "" if loser_min == 10**9 else loser_min,
                "winner_edited_gold_file_and_loser_did_not": int(winner_edited and not loser_edited),
                "winner_reused_prior_test_command": int(winner_test_reuse),
                "winner_reused_prior_failure_signature": int(winner_failure_reuse),
                "loser_followed_irrelevant_file_path": int(loser_irrelevant),
                "loser_produced_empty_patch": int(loser_empty_patch),
                "loser_failed_tests_or_patch_failed": int(loser_failed),
                "winner_patch_files": join(features_by_key.get((iid, c), {}).get("patch_files", "") for c in winners),
                "loser_patch_files": join(features_by_key.get((iid, c), {}).get("patch_files", "") for c in losers),
                "supporting_artifact_paths": join(str(REMOTE_CACHE / iid / c) for c in CONDITIONS),
            }
        )
    return rows


def evidence_excerpt(inv: dict[str, Any], evo: dict[str, Any]) -> str:
    parts = []
    for key in [
        "prior_files_inspected",
        "prior_files_edited",
        "prior_test_commands",
        "error_failure_assertion_lines",
        "grep_search_anchors",
        "patch_shape_hints",
    ]:
        if inv.get(key):
            parts.append(f"{key}: {inv[key]}")
    if evo.get("failure_signature_overlap_excerpt"):
        parts.append(f"failure_overlap: {evo['failure_signature_overlap_excerpt']}")
    return cap(" || ".join(parts), 700)


def add_attr(
    out: list[dict[str, Any]],
    iid: str,
    label: str,
    conditions: list[str],
    contrasting: list[str],
    confidence: str,
    rationale: str,
    excerpts: str,
) -> None:
    out.append(
        {
            "instance_id": iid,
            "mechanism_label": label,
            "supporting_conditions": ";".join(conditions),
            "contrasting_conditions": ";".join(contrasting),
            "confidence": confidence,
            "rationale": rationale,
            "supporting_artifact_paths": join(str(REMOTE_CACHE / iid / c) for c in set(conditions + contrasting)),
            "capped_excerpts": cap(excerpts, 900),
        }
    )


def build_mechanism_attribution(
    valid_rows: list[dict[str, Any]],
    inventory_by_key: dict[tuple[str, str], dict[str, Any]],
    evidence_by_key: dict[tuple[str, str], dict[str, Any]],
    features_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in valid_rows:
        iid = row["instance_id"]
        prior_winners = [c for c in PRIOR_CONDITIONS if row[f"resolved_{c}"]]
        prior_losers = [c for c in PRIOR_CONDITIONS if not row[f"resolved_{c}"]]
        no_solved = bool(row["resolved_no_memory"])
        if not prior_winners and not (no_solved and not any(row[f"resolved_{c}"] for c in PRIOR_CONDITIONS)):
            continue
        added = 0
        for cond in prior_winners:
            inv = inventory_by_key.get((iid, cond), {})
            evo = evidence_by_key.get((iid, cond), {})
            feat = features_by_key.get((iid, cond), {})
            excerpt = evidence_excerpt(inv, evo)
            if safe_int(evo.get("evidence_mentions_target_gold_file_bool")) and (
                safe_int(feat.get("patch_touches_gold_file_bool")) or safe_int(feat.get("edited_target_gold_file_bool"))
            ):
                add_attr(
                    out,
                    iid,
                    "same_file_transfer",
                    [cond],
                    prior_losers + ["no_memory"],
                    "high",
                    "Consistent with prior context exposing the target gold file and the successful run modifying that file.",
                    excerpt,
                )
                added += 1
            elif safe_int(evo.get("evidence_mentions_target_gold_dir_bool")) and (
                safe_int(feat.get("patch_touches_gold_dir_bool")) or safe_int(feat.get("edited_target_gold_dir_bool"))
            ):
                add_attr(
                    out,
                    iid,
                    "same_directory_transfer",
                    [cond],
                    prior_losers + ["no_memory"],
                    "medium",
                    "Consistent with prior context localizing the agent to the target directory rather than an exact same file.",
                    excerpt,
                )
                added += 1
            if safe_int(evo.get("evidence_mentions_target_gold_file_bool")) or safe_int(evo.get("evidence_mentions_target_gold_dir_bool")):
                if feat.get("first_gold_file_read_step") != "":
                    add_attr(
                        out,
                        iid,
                        "localization_hint",
                        [cond],
                        prior_losers + ["no_memory"],
                        "medium",
                        "Consistent with prior-context localization: the winning run reached a target gold file or directory.",
                        excerpt + " || first_gold_file_read_step=" + str(feat.get("first_gold_file_read_step")),
                    )
                    added += 1
            if safe_int(feat.get("ran_prior_test_command_bool")):
                add_attr(
                    out,
                    iid,
                    "test_command_transfer",
                    [cond],
                    prior_losers + ["no_memory"],
                    "high",
                    "Consistent with reuse of a prior-context test or reproduction command.",
                    excerpt + " || run_tests=" + str(feat.get("test_commands_run", "")),
                )
                added += 1
            if safe_int(evo.get("evidence_mentions_target_failure_signature_bool")) or safe_int(feat.get("reproduced_failure_bool")):
                add_attr(
                    out,
                    iid,
                    "failure_signature_transfer",
                    [cond],
                    prior_losers + ["no_memory"],
                    "medium",
                    "Consistent with the run observing a failure signature that overlaps prior or target issue evidence.",
                    excerpt + " || run_failures=" + str(feat.get("failure_signature_excerpts", "")),
                )
                added += 1
            if inv.get("patch_shape_hints") and (
                safe_int(feat.get("patch_touches_gold_file_bool")) or safe_int(feat.get("patch_touches_gold_dir_bool"))
            ):
                add_attr(
                    out,
                    iid,
                    "patch_shape_transfer",
                    [cond],
                    prior_losers + ["no_memory"],
                    "medium",
                    "Consistent with prior-context patch-shape hints aligning with the successful patch location.",
                    excerpt,
                )
                added += 1
            if inv.get("command_sequence_hints") and safe_int(feat.get("num_search_commands")) and safe_int(feat.get("num_file_read_commands")):
                add_attr(
                    out,
                    iid,
                    "procedural_order_transfer",
                    [cond],
                    prior_losers + ["no_memory"],
                    "low",
                    "Consistent with a prior search/read/edit procedure, but not enough to claim causality.",
                    excerpt,
                )
                added += 1
        prior_sum = sum(row[f"resolved_{c}"] for c in PRIOR_CONDITIONS)
        if prior_sum == 1:
            only = prior_winners[0]
            only_score = safe_float(evidence_by_key.get((iid, only), {}).get("evidence_overlap_score"))
            loser_scores = [safe_float(evidence_by_key.get((iid, c), {}).get("evidence_overlap_score")) for c in prior_losers]
            loser_max = max([x for x in loser_scores if not math.isnan(x)] or [0.0])
            if only == "raw":
                label = "raw_redundancy_helped"
                conf = "medium" if only_score >= loser_max else "low"
                add_attr(
                    out,
                    iid,
                    label,
                    [only],
                    prior_losers,
                    conf,
                    "Consistent with raw transcript redundancy preserving useful evidence for the only successful prior representation.",
                    evidence_excerpt(inventory_by_key.get((iid, only), {}), evidence_by_key.get((iid, only), {})),
                )
                added += 1
                for loser in prior_losers:
                    if safe_int(evidence_by_key.get((iid, only), {}).get("evidence_mentions_target_gold_file_bool")) and not safe_int(
                        evidence_by_key.get((iid, loser), {}).get("evidence_mentions_target_gold_file_bool")
                    ):
                        add_attr(
                            out,
                            iid,
                            "memory_extractor_dropped_key_evidence" if loser == "memory" else "ADP_normalization_hurt",
                            [only],
                            [loser],
                            "medium",
                            f"Consistent with {loser} missing target-gold evidence present in raw.",
                            evidence_excerpt(inventory_by_key.get((iid, only), {}), evidence_by_key.get((iid, only), {})),
                        )
                        added += 1
            elif only == "memory":
                add_attr(
                    out,
                    iid,
                    "memory_compression_helped",
                    [only],
                    prior_losers,
                    "medium",
                    "Consistent with compressed memory surfacing useful evidence that did not lead to success in raw/ADP.",
                    evidence_excerpt(inventory_by_key.get((iid, only), {}), evidence_by_key.get((iid, only), {})),
                )
                added += 1
                if features_by_key.get((iid, "raw"), {}).get("first_gold_file_read_step") == "":
                    add_attr(
                        out,
                        iid,
                        "raw_noise_hurt",
                        [only],
                        ["raw"],
                        "low",
                        "Consistent with the raw condition failing to reach the gold file while memory succeeded.",
                        evidence_excerpt(inventory_by_key.get((iid, only), {}), evidence_by_key.get((iid, only), {})),
                    )
                    added += 1
            elif only == "adp":
                add_attr(
                    out,
                    iid,
                    "ADP_structure_helped",
                    [only],
                    prior_losers,
                    "medium",
                    "Consistent with normalized action/observation structure surfacing useful procedural evidence.",
                    evidence_excerpt(inventory_by_key.get((iid, only), {}), evidence_by_key.get((iid, only), {})),
                )
                added += 1
        if not prior_winners and no_solved:
            add_attr(
                out,
                iid,
                "raw_noise_hurt",
                ["no_memory"],
                PRIOR_CONDITIONS,
                "low",
                "Consistent with prior context distracting the agent: no prior representation solved while no_memory did.",
                "No prior-condition solve; inspect representation_disagreement_cases for run behavior.",
            )
            added += 1
        if added == 0 and prior_winners:
            add_attr(
                out,
                iid,
                "no_clear_prior_use",
                prior_winners,
                prior_losers + ["no_memory"],
                "low",
                "Prior representation solved, but extracted evidence and tool behavior do not clearly indicate prior-context use.",
                "",
            )
    return out


def aggregate_tool_patterns(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        groups[(row["condition"], safe_int(row["resolved"]))].append(row)
    rows = []
    for (cond, resolved), items in sorted(groups.items()):
        rows.append(
            {
                "condition": cond,
                "resolved": resolved,
                "n_runs": len(items),
                "avg_num_tool_calls": round(avg(safe_float(r["num_tool_calls"]) for r in items), 3),
                "avg_num_bash_commands": round(avg(safe_float(r["num_bash_commands"]) for r in items), 3),
                "avg_num_search_commands": round(avg(safe_float(r["num_search_commands"]) for r in items), 3),
                "avg_num_file_read_commands": round(avg(safe_float(r["num_file_read_commands"]) for r in items), 3),
                "avg_num_edit_commands": round(avg(safe_float(r["num_edit_commands"]) for r in items), 3),
                "avg_num_test_commands": round(avg(safe_float(r["num_test_commands"]) for r in items), 3),
                "avg_first_gold_file_read_step": round(avg(safe_float_or_none(r["first_gold_file_read_step"]) for r in items), 3),
                "rate_patch_touches_gold_file": round(avg(safe_float(r["patch_touches_gold_file_bool"]) for r in items), 3),
                "rate_ran_prior_test_command": round(avg(safe_float(r["ran_prior_test_command_bool"]) for r in items), 3),
                "rate_reproduced_failure": round(avg(safe_float(r["reproduced_failure_bool"]) for r in items), 3),
            }
        )
    return rows


def safe_float_or_none(x: Any) -> float:
    if x == "" or x is None:
        return float("nan")
    return safe_float(x)


def avg(values: Iterable[float]) -> float:
    xs = [x for x in values if not math.isnan(x)]
    return statistics.mean(xs) if xs else float("nan")


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return ""
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def mechanism_counts(attrib: list[dict[str, Any]], instance_filter: set[str] | None = None) -> Counter:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in attrib:
        if instance_filter is None or row["instance_id"] in instance_filter:
            seen[row["mechanism_label"]].add(row["instance_id"])
    return Counter({label: len(ids) for label, ids in seen.items()})


def write_reports(
    summary: dict[str, Any],
    valid_rows: list[dict[str, Any]],
    features: list[dict[str, Any]],
    tool_patterns: list[dict[str, Any]],
    evidence_overlap: list[dict[str, Any]],
    inventory_by_key: dict[tuple[str, str], dict[str, Any]],
    attribution: list[dict[str, Any]],
    disagreement: list[dict[str, Any]],
) -> None:
    prior_oracle_ids = {r["instance_id"] for r in valid_rows if r["oracle_prior"]}
    raw_only = {r["instance_id"] for r in valid_rows if r["resolved_raw"] and not r["resolved_adp"] and not r["resolved_memory"]}
    adp_only = {r["instance_id"] for r in valid_rows if r["resolved_adp"] and not r["resolved_raw"] and not r["resolved_memory"]}
    memory_only = {r["instance_id"] for r in valid_rows if r["resolved_memory"] and not r["resolved_raw"] and not r["resolved_adp"]}
    n = summary["n"]

    first_gold_rows = []
    for cond in CONDITIONS:
        for resolved in [0, 1]:
            items = [r for r in features if r["condition"] == cond and safe_int(r["resolved"]) == resolved]
            first_gold_rows.append(
                {
                    "condition": cond,
                    "resolved": resolved,
                    "n": len(items),
                    "avg_first_gold_file_read_step": round(avg(safe_float_or_none(r["first_gold_file_read_step"]) for r in items), 2),
                }
            )
    patch_touch_rows = []
    for cond in CONDITIONS:
        items = [r for r in features if r["condition"] == cond]
        patch_touch_rows.append(
            {
                "condition": cond,
                "rate_patch_touches_gold_file": round(avg(safe_float(r["patch_touches_gold_file_bool"]) for r in items), 3),
                "rate_patch_touches_gold_dir": round(avg(safe_float(r["patch_touches_gold_dir_bool"]) for r in items), 3),
            }
        )
    successful = [r for r in features if safe_int(r["resolved"])]
    prior_touch_rate = avg(
        1.0 if safe_int(r["touched_prior_inspected_file_bool"]) or safe_int(r["touched_prior_edited_file_bool"]) else 0.0
        for r in successful
    )

    def count_lines(counter: Counter) -> str:
        if not counter:
            return "- none"
        return "\n".join(f"- {k}: {v}" for k, v in counter.most_common())

    raw_absent_memory_examples = []
    memory_examples = []
    adp_examples = []
    distraction_examples = []
    for iid in sorted(raw_only):
        raw_inv = inventory_by_key.get((iid, "raw"), {})
        mem_inv = inventory_by_key.get((iid, "memory"), {})
        if raw_inv and mem_inv:
            raw_absent_memory_examples.append(
                f"- {iid}: raw files/tests include `{cap(raw_inv.get('prior_files_mentioned'), 180)}`; "
                f"memory files/tests include `{cap(mem_inv.get('prior_files_mentioned'), 120)}`."
            )
    for iid in sorted(memory_only):
        inv = inventory_by_key.get((iid, "memory"), {})
        memory_examples.append(f"- {iid}: memory excerpt `{cap(evidence_excerpt(inv, {}), 260)}`.")
    for iid in sorted(adp_only):
        inv = inventory_by_key.get((iid, "adp"), {})
        adp_examples.append(f"- {iid}: ADP excerpt `{cap(inv.get('command_sequence_hints'), 260)}`.")
    for d in disagreement:
        if "no_memory_solved_all_prior_failed" in d["case_types"]:
            distraction_examples.append(
                f"- {d['instance_id']}: no_memory solved while prior conditions failed; "
                f"loser_empty_patch={d['loser_produced_empty_patch']}, loser_failed={d['loser_failed_tests_or_patch_failed']}."
            )

    report = f"""# Oracle Mechanism Report

## Validation

- Valid targets: {n}
- Excluded target: {EXCLUDED}
- no_memory: {summary['counts']['no_memory']} / {n}
- raw: {summary['counts']['raw']} / {n}
- adp: {summary['counts']['adp']} / {n}
- memory: {summary['counts']['memory']} / {n}
- oracle_prior: {summary['oracle_prior']} / {n} ({summary['oracle_prior'] / n:.3f})
- oracle_all: {summary['oracle_all']} / {n} ({summary['oracle_all'] / n:.3f})

The prior-context oracle is retrospective and not deployable. The mechanism labels below mean "consistent with" the transcript evidence, not causal proof.

## Representation-Choice Headroom

- raw-only prior solves: {len(raw_only)}
- memory-only prior solves: {len(memory_only)}
- ADP-only prior solves: {len(adp_only)}
- prior oracle solves beyond best fixed prior representation: {summary['oracle_prior'] - max(summary['counts'][c] for c in PRIOR_CONDITIONS)}
- oracle_all gain over prior-only oracle: {summary['oracle_all'] - summary['oracle_prior']}

## Mechanism Counts

Among prior-oracle solves:
{count_lines(mechanism_counts(attribution, prior_oracle_ids))}

Among raw-only solves:
{count_lines(mechanism_counts(attribution, raw_only))}

Among memory-only solves:
{count_lines(mechanism_counts(attribution, memory_only))}

Among ADP-only solves:
{count_lines(mechanism_counts(attribution, adp_only))}

## Time To Gold File

{md_table(first_gold_rows, ['condition', 'resolved', 'n', 'avg_first_gold_file_read_step'])}

## Tool Patterns By Outcome

{md_table(tool_patterns, ['condition', 'resolved', 'n_runs', 'avg_num_search_commands', 'avg_num_test_commands', 'avg_num_edit_commands', 'rate_patch_touches_gold_file'])}

## Patch Touch Rates

{md_table(patch_touch_rows, ['condition', 'rate_patch_touches_gold_file', 'rate_patch_touches_gold_dir'])}

Successful runs that touched a prior-inspected or prior-edited file: {prior_touch_rate:.3f}.

## Representation-Specific Examples

Examples of evidence present in raw but absent or reduced in memory:
{os.linesep.join(raw_absent_memory_examples[:6]) or '- none found by heuristic'}

Examples where memory compression appears helpful:
{os.linesep.join(memory_examples[:6]) or '- none found by heuristic'}

Examples where ADP structure appears helpful:
{os.linesep.join(adp_examples[:6]) or '- none found by heuristic'}

Examples where prior context is consistent with distraction:
{os.linesep.join(distraction_examples[:6]) or '- none found by heuristic'}

## Paper Framing

The prior-context oracle shows representation-choice headroom over any fixed representation. The transcript evidence is most consistent with localization hints, test/failure transfer, patch-shape transfer, and representation-specific noise or compression effects. These logs support mechanism attribution hypotheses; they do not prove causal mechanisms without ablation.
"""
    (REPORTS / "oracle_mechanism_report.md").write_text(report, encoding="utf-8")

    case_counter = Counter()
    for d in disagreement:
        for ct in d["case_types"].split(";"):
            case_counter[ct] += 1
    case_lines = ["# Representation Disagreement Case Studies", "", "## Case Counts", ""]
    case_lines.extend(f"- {k}: {v}" for k, v in case_counter.most_common())
    case_lines.append("")
    case_lines.append("## Cases")
    for d in disagreement:
        case_lines.append("")
        case_lines.append(f"### {d['instance_id']}")
        case_lines.append("")
        case_lines.append(f"- Types: {d['case_types']}")
        case_lines.append(f"- Outcomes: {d['condition_outcomes']}")
        case_lines.append(f"- Winning conditions: {d['winning_conditions'] or 'none'}")
        case_lines.append(f"- Evidence in winner missing from loser: {d['evidence_present_in_winner_missing_from_loser'] or 'none'}")
        case_lines.append(
            "- Behavior: "
            f"winner_earlier_gold={d['winner_opened_gold_file_earlier']}, "
            f"winner_edited_gold={d['winner_edited_gold_file_and_loser_did_not']}, "
            f"test_reuse={d['winner_reused_prior_test_command']}, "
            f"failure_reuse={d['winner_reused_prior_failure_signature']}, "
            f"loser_empty_patch={d['loser_produced_empty_patch']}, "
            f"loser_failed={d['loser_failed_tests_or_patch_failed']}"
        )
        case_lines.append(f"- Artifacts: `{d['supporting_artifact_paths']}`")
    (REPORTS / "representation_disagreement_case_studies.md").write_text("\n".join(case_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-remote", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.fetch_remote or args.refresh:
        fetch_remote_artifacts(refresh=args.refresh)
    if not REMOTE_CACHE.exists():
        raise SystemExit(f"Missing artifact mirror {REMOTE_CACHE}; rerun with --fetch-remote and SSHPASS set.")

    paired_rows, summary = validate_oracles(read_csv(DATA / "paired_results.csv"))
    overlap_rows = read_csv(DATA / "overlap_features.csv")
    overlap_by_id = {r["instance_id"]: r for r in overlap_rows if r["instance_id"] != EXCLUDED}

    timeline = build_tool_timeline(paired_rows)
    inventory, inventory_by_key, target_info = build_prompt_indices(paired_rows, overlap_by_id)
    evidence_overlap, evidence_by_key = build_evidence_overlap(inventory, overlap_by_id, target_info)
    features, features_by_key, test_reuse, failure_reuse = build_behavior_features(
        paired_rows, timeline, overlap_by_id, inventory_by_key, target_info
    )
    disagreement = build_disagreement_cases(paired_rows, evidence_by_key, features_by_key)
    attribution = build_mechanism_attribution(paired_rows, inventory_by_key, evidence_by_key, features_by_key)
    tool_patterns = aggregate_tool_patterns(features)

    write_csv(DATA / "tool_timeline_events.csv", timeline)
    write_csv(DATA / "representation_evidence_inventory.csv", inventory)
    write_csv(DATA / "evidence_target_overlap.csv", evidence_overlap)
    write_csv(DATA / "oracle_mechanism_attribution.csv", attribution)
    write_csv(DATA / "representation_disagreement_cases.csv", disagreement)
    write_csv(DATA / "tool_pattern_by_outcome.csv", tool_patterns)
    write_csv(DATA / "time_to_gold_file.csv", features)
    write_csv(DATA / "test_command_reuse.csv", test_reuse)
    write_csv(DATA / "failure_signature_reuse.csv", failure_reuse)
    write_reports(summary, paired_rows, features, tool_patterns, evidence_overlap, inventory_by_key, attribution, disagreement)

    print(
        "Wrote mechanism analysis outputs: "
        f"{len(timeline)} timeline rows, {len(features)} run feature rows, "
        f"{len(attribution)} attribution rows, {len(disagreement)} disagreement cases."
    )


if __name__ == "__main__":
    main()
