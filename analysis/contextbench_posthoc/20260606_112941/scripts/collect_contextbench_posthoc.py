#!/usr/bin/env python3
"""Collect ContextBench posthoc artifacts into target-level analysis tables.

This script is intentionally conservative:
  * evaluator outputs are the only source used for resolved_* fields;
  * generated run artifacts only enrich run/patch/token metadata;
  * missing evaluator/run data is left blank and documented.

Remote discovery is supported when SSHPASS is set, but disabled by default so a
normal rerun never prompts for credentials or writes to the remote host.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CONDITIONS = ["no_memory", "raw", "adp", "memory"]
EXCLUDED_TARGET = "django__django-28147"
SYMPY_ANOMALY = "sympy__sympy-19006"
EXPECTED_COUNTS = {"no_memory": 10, "raw": 19, "adp": 15, "memory": 16}

REMOTE_ROOTS = [
    "/mnt/data/shashwat/memory-representation",
    "/mnt/data/shashwat/openhands-adp-memory",
]
EXPECTED_REMOTE_EXECUTION_SUFFIX = "data/contextbench_phase2/execution_full_qwen36_65k_fix1"


PAIRED_COLUMNS = [
    "instance_id",
    "repo",
    "prior_instance_id",
    "target_instance_id",
    "base_commit",
    "condition_set_complete",
    "resolved_no_memory",
    "resolved_raw",
    "resolved_adp",
    "resolved_memory",
    "eval_status_no_memory",
    "eval_status_raw",
    "eval_status_adp",
    "eval_status_memory",
    "eval_error_no_memory",
    "eval_error_raw",
    "eval_error_adp",
    "eval_error_memory",
    "empty_patch_no_memory",
    "empty_patch_raw",
    "empty_patch_adp",
    "empty_patch_memory",
    "non_empty_patch_no_memory",
    "non_empty_patch_raw",
    "non_empty_patch_adp",
    "non_empty_patch_memory",
    "patch_failed_no_memory",
    "patch_failed_raw",
    "patch_failed_adp",
    "patch_failed_memory",
    "no_patch_eval_no_memory",
    "no_patch_eval_raw",
    "no_patch_eval_adp",
    "no_patch_eval_memory",
    "patch_bytes_no_memory",
    "patch_bytes_raw",
    "patch_bytes_adp",
    "patch_bytes_memory",
    "patch_lines_added_no_memory",
    "patch_lines_added_raw",
    "patch_lines_added_adp",
    "patch_lines_added_memory",
    "patch_lines_deleted_no_memory",
    "patch_lines_deleted_raw",
    "patch_lines_deleted_adp",
    "patch_lines_deleted_memory",
    "patch_files_changed_no_memory",
    "patch_files_changed_raw",
    "patch_files_changed_adp",
    "patch_files_changed_memory",
    "run_exit_code_no_memory",
    "run_exit_code_raw",
    "run_exit_code_adp",
    "run_exit_code_memory",
    "run_status_no_memory",
    "run_status_raw",
    "run_status_adp",
    "run_status_memory",
    "openhands_iterations_no_memory",
    "openhands_iterations_raw",
    "openhands_iterations_adp",
    "openhands_iterations_memory",
    "llm_calls_no_memory",
    "llm_calls_raw",
    "llm_calls_adp",
    "llm_calls_memory",
    "input_tokens_no_memory",
    "input_tokens_raw",
    "input_tokens_adp",
    "input_tokens_memory",
    "output_tokens_no_memory",
    "output_tokens_raw",
    "output_tokens_adp",
    "output_tokens_memory",
    "total_tokens_no_memory",
    "total_tokens_raw",
    "total_tokens_adp",
    "total_tokens_memory",
    "max_turn_tokens_no_memory",
    "max_turn_tokens_raw",
    "max_turn_tokens_adp",
    "max_turn_tokens_memory",
    "wall_seconds_no_memory",
    "wall_seconds_raw",
    "wall_seconds_adp",
    "wall_seconds_memory",
    "prompt_chars_no_memory",
    "prompt_chars_raw",
    "prompt_chars_adp",
    "prompt_chars_memory",
    "prior_context_chars_no_memory",
    "prior_context_chars_raw",
    "prior_context_chars_adp",
    "prior_context_chars_memory",
    "empty_patch_cause_no_memory",
    "empty_patch_cause_raw",
    "empty_patch_cause_adp",
    "empty_patch_cause_memory",
    "notes",
]


CONDITION_SUMMARY_COLUMNS = [
    "condition",
    "n_targets",
    "resolved_count",
    "success_rate",
    "wilson_95_low",
    "wilson_95_high",
    "non_empty_patch_count",
    "empty_patch_count",
    "patch_attempt_rate",
    "eval_no_patch_count",
    "eval_patch_failed_count",
    "other_eval_error_count",
    "mean_wall_seconds",
    "median_wall_seconds",
    "total_wall_seconds",
    "mean_llm_calls",
    "median_llm_calls",
    "total_llm_calls",
    "mean_input_tokens",
    "median_input_tokens",
    "total_input_tokens",
    "mean_output_tokens",
    "median_output_tokens",
    "total_output_tokens",
    "mean_total_tokens",
    "median_total_tokens",
    "total_total_tokens",
    "median_max_turn_tokens",
    "max_max_turn_tokens",
    "mean_patch_bytes",
    "median_patch_bytes",
    "mean_patch_files_changed",
    "median_patch_files_changed",
]


PATCH_QUALITY_COLUMNS = [
    "condition",
    "resolved_count",
    "non_empty_patch_count",
    "resolved_given_non_empty_patch_rate",
    "empty_patch_count",
    "unresolved_non_empty_patch_count",
]


RUN_STATUS_COLUMNS = [
    "instance_id",
    "run_status_no_memory",
    "run_status_raw",
    "run_status_adp",
    "run_status_memory",
    "run_exit_code_no_memory",
    "run_exit_code_raw",
    "run_exit_code_adp",
    "run_exit_code_memory",
    "wall_seconds_no_memory",
    "wall_seconds_raw",
    "wall_seconds_adp",
    "wall_seconds_memory",
]


EVAL_STATUS_COLUMNS = [
    "instance_id",
    "eval_status_no_memory",
    "eval_status_raw",
    "eval_status_adp",
    "eval_status_memory",
    "eval_error_no_memory",
    "eval_error_raw",
    "eval_error_adp",
    "eval_error_memory",
]


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if isinstance(obj, dict):
                obj["_source_path"] = str(path)
                obj["_source_lineno"] = lineno
                rows.append(obj)
    return rows


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_bool_int(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value) if isinstance(value, float) else False:
            return ""
        if value in (0, 1):
            return str(int(value))
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes", "resolved", "pass", "passed", "success", "successfully_resolved"}:
            return "1"
        if low in {"false", "0", "no", "unresolved", "fail", "failed", "error", "not_resolved"}:
            return "0"
    return ""


def to_float(value: Any) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, bool):
        return str(int(value))
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return ""


def to_int(value: Any) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, bool):
        return str(int(value))
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        v = row.get(key, "")
        if v in ("", None):
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def mean(values: list[float]) -> str:
    return "" if not values else f"{statistics.fmean(values):.6g}"


def median(values: list[float]) -> str:
    return "" if not values else f"{statistics.median(values):.6g}"


def total(values: list[float]) -> str:
    return "" if not values else f"{sum(values):.6g}"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return center - half, center + half


def stable_hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_relevant_files(root: Path, *, max_files: int | None = None) -> list[tuple[str, int]]:
    """List relevant files while skipping heavyweight workspaces and dependency dirs."""
    if not root.exists():
        return []
    relevant_names = {
        "prompt.txt",
        "stdout.jsonl",
        "stderr.log",
        "run_meta.json",
        "patch.diff",
        "prediction.json",
        "prediction_audit.json",
        "command.json",
        "base_state.json",
        "run_manifest.jsonl",
        "pair_manifest.jsonl",
        "prompt_manifest.jsonl",
        "prompt_render_report.json",
        "forbidden_prompt_scan.txt",
        "smoke_run_report.json",
        "smoke_run_plan.json",
    }
    relevant_patterns = (
        "_preds.json",
        "_predictions.jsonl",
        "predictions.jsonl",
        "results.json",
        "report.json",
        "eval",
        "keepimg",
        "qwen36",
        "token",
        "conversation",
    )
    skip_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        "workspace",
        ".mypy_cache",
        ".pytest_cache",
    }
    out: list[tuple[str, int]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            low = name.lower()
            path = Path(dirpath) / name
            if name in relevant_names or any(p in low for p in relevant_patterns):
                try:
                    size = path.stat().st_size
                except OSError:
                    size = -1
                out.append((str(path), size))
                if max_files and len(out) >= max_files:
                    return out
    return sorted(out)


def ssh_remote_discovery() -> tuple[list[str], str]:
    """Run read-only remote discovery if SSHPASS is present.

    The remote command only traverses the two approved roots. It prunes workspace
    directories and prints paths/sizes for metadata files.
    """
    if not os.environ.get("SSHPASS"):
        return [], "SSHPASS not set; remote discovery not attempted by script."
    remote_script = r'''
set -eu
for root in /mnt/data/shashwat/memory-representation /mnt/data/shashwat/openhands-adp-memory; do
  if [ -d "$root" ]; then
    printf 'ROOT_EXISTS\t%s\n' "$root"
    expected="$root/data/contextbench_phase2/execution_full_qwen36_65k_fix1"
    if [ -d "$expected" ]; then printf 'EXPECTED_EXECUTION_DIR\t%s\n' "$expected"; fi
    find "$root" \
      \( -name workspace -o -name .git -o -name node_modules -o -name __pycache__ \) -prune -o \
      -type f \( \
        -name 'prompt.txt' -o -name 'stdout.jsonl' -o -name 'stderr.log' -o -name 'run_meta.json' -o \
        -name 'patch.diff' -o -name 'prediction.json' -o -name 'prediction_audit.json' -o -name 'command.json' -o \
        -name 'run_manifest.jsonl' -o -name 'pair_manifest.jsonl' -o -name 'prompt_manifest.jsonl' -o \
        -name 'prompt_render_report.json' -o -name 'forbidden_prompt_scan.txt' -o \
        -name '*predictions.jsonl' -o -name '*_preds.json' -o -name '*result*.json' -o -name '*report*.json' -o \
        -name '*token*' \
      \) -printf 'FILE\t%p\t%s\n'
  else
    printf 'ROOT_MISSING\t%s\n' "$root"
  fi
done
'''
    argv = [
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
        remote_script,
    ]
    try:
        proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return [], f"remote discovery exception: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return [], f"remote discovery failed with exit {proc.returncode}: {proc.stderr.strip()[-1000:]}"
    return proc.stdout.splitlines(), "remote discovery completed using SSHPASS/sshpass."


def load_manifests(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    phase2 = repo_root / "data/contextbench_phase2"
    prompt_rows = read_jsonl(phase2 / "prompt_manifest.jsonl")
    pair_rows = read_jsonl(phase2 / "pair_manifest.jsonl")
    run_rows = read_jsonl(phase2 / "run_manifest.jsonl")
    report_path = phase2 / "prompt_render_report.json"
    render_report = read_json(report_path) if report_path.exists() else {}
    return prompt_rows, pair_rows, run_rows, render_report


def group_prompt_manifest(prompt_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in prompt_rows:
        target = str(row.get("target_instance_id", ""))
        cond = str(row.get("condition", ""))
        if target and cond:
            grouped[target][cond] = row
    return grouped


def group_run_manifest(run_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in run_rows:
        target = str(row.get("target_instance_id", ""))
        cond = str(row.get("condition", ""))
        if target and cond:
            grouped[target][cond] = row
    return grouped


def pair_index(pair_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in pair_rows:
        target = str(row.get("related_instance_id", ""))
        if target:
            out[target] = row
    return out


def parse_patch(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    files: set[str] = set()
    added = 0
    deleted = 0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a = parts[2][2:] if parts[2].startswith("a/") else parts[2]
                b = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                for candidate in (b, a):
                    if candidate and candidate != "/dev/null":
                        files.add(candidate)
                        break
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
    return {
        "patch_bytes": str(len(text.encode("utf-8"))),
        "patch_lines_added": str(added),
        "patch_lines_deleted": str(deleted),
        "patch_files_changed": str(len(files)),
        "empty_patch": "1" if len(text.strip()) == 0 else "0",
        "non_empty_patch": "0" if len(text.strip()) == 0 else "1",
    }


def recursive_get(obj: Any, names: set[str]) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in names:
                return value
        for value in obj.values():
            found = recursive_get(value, names)
            if found not in (None, ""):
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = recursive_get(value, names)
            if found not in (None, ""):
                return found
    return None


def parse_run_meta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {"run_status": "run_meta_parse_error"}
    status = recursive_get(data, {"status", "run_status", "state"}) or ""
    exit_code = recursive_get(data, {"exit_code", "returncode", "return_code"}) or ""
    wall = recursive_get(data, {"elapsed_seconds", "wall_seconds", "duration_seconds", "openhands_elapsed_seconds"}) or ""
    iterations = recursive_get(data, {"iterations", "openhands_iterations", "iteration_count"}) or ""
    return {
        "run_status": str(status),
        "run_exit_code": to_int(exit_code),
        "wall_seconds": to_float(wall),
        "openhands_iterations": to_int(iterations),
    }


def parse_stdout_token_summary(path: Path, max_bytes: int = 50_000_000) -> dict[str, str]:
    """Stream stdout JSONL and pull approximate OpenHands token/call counters."""
    if not path.exists():
        return {}
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size > max_bytes:
        # Still stream the tail would need seeking and JSONL boundary handling. Leave
        # blank rather than risking a very slow local analysis.
        return {"run_status": "stdout_too_large_for_default_token_scan"}
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    max_turn = 0
    llm_calls = 0
    iterations = 0
    token_seen = False
    iteration_re = re.compile(r"\biteration\b", re.I)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            text = line
            obj: Any | None = None
            try:
                obj = json.loads(line)
                text = json.dumps(obj, ensure_ascii=False)
            except json.JSONDecodeError:
                obj = None
            if "prompt_tokens" in text or "input_tokens" in text or "completion_tokens" in text or "output_tokens" in text:
                llm_calls += 1
                token_seen = True
                nums = extract_token_numbers(obj if obj is not None else text)
                input_tokens += nums.get("input_tokens", 0)
                output_tokens += nums.get("output_tokens", 0)
                total_tokens += nums.get("total_tokens", 0)
                max_turn = max(max_turn, nums.get("turn_tokens", 0))
            if iteration_re.search(text):
                iterations += 1
    out: dict[str, str] = {}
    if token_seen:
        out.update(
            {
                "llm_calls": str(llm_calls),
                "input_tokens": str(input_tokens),
                "output_tokens": str(output_tokens),
                "total_tokens": str(total_tokens if total_tokens else input_tokens + output_tokens),
                "max_turn_tokens": str(max_turn),
            }
        )
    if iterations:
        out["openhands_iterations"] = str(iterations)
    return out


def extract_token_numbers(obj: Any) -> dict[str, int]:
    names_in = {"input_tokens", "prompt_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"}
    names_out = {"output_tokens", "completion_tokens"}
    names_total = {"total_tokens"}
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "turn_tokens": 0}

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            local_in = 0
            local_out = 0
            local_total = 0
            for k, v in x.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if k in names_in:
                        local_in += int(v)
                    elif k in names_out:
                        local_out += int(v)
                    elif k in names_total:
                        local_total += int(v)
                walk(v)
            if local_in or local_out or local_total:
                totals["input_tokens"] += local_in
                totals["output_tokens"] += local_out
                totals["total_tokens"] += local_total
                totals["turn_tokens"] = max(totals["turn_tokens"], local_total or local_in + local_out)
        elif isinstance(x, list):
            for y in x:
                walk(y)

    if isinstance(obj, str):
        for key, dest in [
            ("input_tokens", "input_tokens"),
            ("prompt_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("completion_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        ]:
            for match in re.finditer(rf'"?{re.escape(key)}"?\s*[:=]\s*(\d+)', obj):
                value = int(match.group(1))
                totals[dest] += value
                totals["turn_tokens"] = max(totals["turn_tokens"], value)
    else:
        walk(obj)
    return totals


def classify_empty_patch_cause(run_dir: Path) -> str:
    text_parts: list[str] = []
    for name in ["run_meta.json", "stderr.log"]:
        path = run_dir / name
        if path.exists():
            try:
                text_parts.append(path.read_text(encoding="utf-8", errors="replace")[:20000])
            except OSError:
                pass
    stdout = run_dir / "stdout.jsonl"
    if stdout.exists():
        try:
            with stdout.open("r", encoding="utf-8", errors="replace") as f:
                for _, line in zip(range(200), f):
                    text_parts.append(line)
        except OSError:
            pass
    text = "\n".join(text_parts).lower()
    if not text:
        return "unknown"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "head" in text and "mismatch" in text:
        return "head_mismatch"
    if "permission denied" in text or "workspace" in text and ("not found" in text or "error" in text):
        return "workspace_error"
    if "empty response" in text or "empty message" in text:
        return "model_empty_response"
    if "security risk" in text and "unknown" in text:
        return "crash"
    if "traceback" in text or "exception" in text or "crash" in text:
        return "crash"
    if text.count("command") > 50 and "git diff" not in text:
        return "tool_loop"
    return "normal_no_edit"


def candidate_run_dirs(repo_root: Path, target: str, condition: str, artifact_roots: list[Path]) -> list[Path]:
    candidates = []
    for root in artifact_roots:
        candidates.append(root / "runs" / target / condition)
        candidates.append(root / target / condition)
    candidates.extend(
        [
            repo_root / "data/contextbench_phase2/execution_full_qwen36_65k_fix1/runs" / target / condition,
            repo_root / "data/contextbench_phase2/execution/runs" / target / condition,
            repo_root / "runs/contextbench_qwen3_30b_openhands" / target / condition,
        ]
    )
    return [p for p in candidates if p.exists()]


def load_run_artifacts(repo_root: Path, target: str, condition: str, artifact_roots: list[Path]) -> dict[str, str]:
    for run_dir in candidate_run_dirs(repo_root, target, condition, artifact_roots):
        data: dict[str, str] = {}
        data.update(parse_patch(run_dir / "patch.diff"))
        data.update(parse_run_meta(run_dir / "run_meta.json"))
        stdout_summary = parse_stdout_token_summary(run_dir / "stdout.jsonl")
        for k, v in stdout_summary.items():
            if v != "":
                data[k] = v
        if data.get("empty_patch") == "1":
            data["empty_patch_cause"] = classify_empty_patch_cause(run_dir)
        data["run_artifact_dir"] = str(run_dir)
        return data
    return {}


def extract_eval_record_from_mapping(obj: Any, path: Path, condition_hint: str | None) -> dict[tuple[str, str], dict[str, str]]:
    """Best-effort parser for SWE-bench evaluator result JSON/JSONL variants."""
    out: dict[tuple[str, str], dict[str, str]] = {}

    def condition_from_path() -> str | None:
        low = str(path).lower()
        for cond in CONDITIONS:
            if f"_{cond}_" in low or f"/{cond}/" in low or f"full_{cond}_" in low or cond in path.name.lower():
                return cond
        return condition_hint

    def add(instance: str, rec: Any, cond: str | None) -> None:
        if not instance or not cond or cond not in CONDITIONS:
            return
        status = ""
        error = ""
        resolved = ""
        if isinstance(rec, dict):
            for key in ["resolved", "is_resolved", "success", "passed", "PASS", "result"]:
                if key in rec:
                    resolved = as_bool_int(rec.get(key))
                    if resolved:
                        break
            for key in ["status", "eval_status", "resolution_status", "result"]:
                if key in rec and rec.get(key) is not None:
                    status = str(rec.get(key))
                    if not resolved:
                        resolved = as_bool_int(status)
                    break
            for key in ["error", "eval_error", "failure", "message", "traceback"]:
                if key in rec and rec.get(key):
                    error = str(rec.get(key))[:500]
                    break
        else:
            resolved = as_bool_int(rec)
            status = str(rec)
        out[(instance, cond)] = {
            "resolved": resolved,
            "eval_status": status,
            "eval_error": error,
            "eval_source_path": str(path),
        }

    cond = condition_from_path()
    if isinstance(obj, dict):
        # Common form: {"instance_id": {...}}.
        for key, value in obj.items():
            if isinstance(key, str) and "__" in key and "-" in key:
                add(key, value, cond)
        # Common form: {"instance_id": "...", "resolved": true}.
        instance = obj.get("instance_id") or obj.get("target_instance_id")
        row_cond = obj.get("condition") or cond
        if instance:
            add(str(instance), obj, str(row_cond) if row_cond else cond)
        # Common nested containers.
        for container_key in ["results", "resolved", "instances", "report"]:
            val = obj.get(container_key)
            if isinstance(val, list):
                for item in val:
                    nested = extract_eval_record_from_mapping(item, path, cond)
                    out.update(nested)
            elif isinstance(val, dict):
                nested = extract_eval_record_from_mapping(val, path, cond)
                out.update(nested)
    elif isinstance(obj, list):
        for item in obj:
            nested = extract_eval_record_from_mapping(item, path, cond)
            out.update(nested)
    return out


def discover_eval_records(repo_root: Path, extra_roots: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    roots = [
        repo_root / "data/contextbench_phase2",
        repo_root / "reports",
    ] + extra_roots
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {".git", "workspace", "__pycache__", ".venv"}]
            for name in filenames:
                low = name.lower()
                if not (low.endswith(".json") or low.endswith(".jsonl")):
                    continue
                if any(s in low for s in ["result", "report", "eval", "keepimg"]) or "predictions" in str(Path(dirpath)).lower():
                    candidates.append(Path(dirpath) / name)
    out: dict[tuple[str, str], dict[str, str]] = {}
    for path in candidates:
        # Do not treat prediction/model_patch files as evaluator truth.
        low = path.name.lower()
        if "prediction" in low or low.endswith("_preds.json"):
            continue
        try:
            if low.endswith(".jsonl"):
                for row in read_jsonl(path):
                    out.update(extract_eval_record_from_mapping(row, path, None))
            else:
                out.update(extract_eval_record_from_mapping(read_json(path), path, None))
        except Exception:
            continue
    return out


def build_paired_rows(
    repo_root: Path,
    prompt_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    artifact_roots: list[Path],
    eval_records: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    prompt_by_target = group_prompt_manifest(prompt_rows)
    run_by_target = group_run_manifest(run_rows)
    pairs = pair_index(pair_rows)
    targets = sorted(t for t in prompt_by_target if t != EXCLUDED_TARGET)
    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    for target in targets:
        conds = prompt_by_target.get(target, {})
        pair = pairs.get(target, {})
        representative = next(iter(conds.values()), {})
        row: dict[str, Any] = {col: "" for col in PAIRED_COLUMNS}
        row["instance_id"] = target
        row["target_instance_id"] = target
        row["repo"] = representative.get("target_repo") or pair.get("repo") or ""
        row["prior_instance_id"] = representative.get("prior_instance_id") or pair.get("experience_instance_id") or ""
        row["base_commit"] = representative.get("target_base_commit") or pair.get("target_base_commit") or ""
        row["condition_set_complete"] = "1" if set(conds) >= set(CONDITIONS) else "0"
        row_notes: list[str] = []
        if row["condition_set_complete"] != "1":
            row_notes.append(f"missing prompt conditions: {sorted(set(CONDITIONS) - set(conds))}")
        for cond in CONDITIONS:
            prompt = conds.get(cond, {})
            run_manifest_row = run_by_target.get(target, {}).get(cond, {})
            if prompt:
                row[f"prompt_chars_{cond}"] = prompt.get("prompt_chars", "")
                row[f"prior_context_chars_{cond}"] = prompt.get("prior_context_chars", "")
            elif run_manifest_row:
                row[f"prompt_chars_{cond}"] = ""
                row[f"prior_context_chars_{cond}"] = ""
            eval_rec = eval_records.get((target, cond), {})
            if eval_rec:
                row[f"resolved_{cond}"] = eval_rec.get("resolved", "")
                row[f"eval_status_{cond}"] = eval_rec.get("eval_status", "")
                row[f"eval_error_{cond}"] = eval_rec.get("eval_error", "")
                status_low = (row[f"eval_status_{cond}"] + " " + row[f"eval_error_{cond}"]).lower()
                row[f"patch_failed_{cond}"] = "1" if "patch" in status_low and "fail" in status_low else "0"
                row[f"no_patch_eval_{cond}"] = "1" if "no patch" in status_low or "empty patch" in status_low else "0"
            run_art = load_run_artifacts(repo_root, target, cond, artifact_roots)
            if run_art:
                for key, value in run_art.items():
                    if key in {"run_artifact_dir"}:
                        continue
                    row[f"{key}_{cond}"] = value
            # Conservative fallback: if patch.diff is known absent, do not infer
            # empty_patch; absence could mean artifacts are missing.
        if not any(row.get(f"resolved_{cond}") in {"0", "1"} for cond in CONDITIONS):
            row_notes.append("evaluator outputs not found for this target")
        row["notes"] = "; ".join(row_notes)
        rows.append(row)

    if len(rows) != 95:
        notes.append(f"paired_results row count is {len(rows)}, expected 95 after excluding {EXCLUDED_TARGET}.")
    if any(r["instance_id"] == EXCLUDED_TARGET for r in rows):
        notes.append(f"excluded target {EXCLUDED_TARGET} is present in paired_results.")
    for cond in CONDITIONS:
        values = [r.get(f"resolved_{cond}", "") for r in rows]
        if all(v in {"0", "1"} for v in values):
            count = sum(int(v) for v in values)
            if count != EXPECTED_COUNTS[cond]:
                notes.append(f"{cond} resolved count {count} does not match expected {EXPECTED_COUNTS[cond]}.")
        else:
            missing = sum(1 for v in values if v not in {"0", "1"})
            notes.append(
                f"{cond} resolved count cannot be validated: {missing}/{len(values)} rows lack evaluator truth."
            )
    return rows, notes


def summarize_conditions(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    condition_rows: list[dict[str, Any]] = []
    patch_quality: list[dict[str, Any]] = []
    for cond in CONDITIONS:
        n = len(rows)
        resolved_values = [r.get(f"resolved_{cond}", "") for r in rows]
        resolved_complete = all(v in {"0", "1"} for v in resolved_values)
        resolved_count = sum(int(v) for v in resolved_values if v in {"0", "1"})
        low, high = wilson_ci(resolved_count, n) if resolved_complete else (float("nan"), float("nan"))
        non_empty_values = [r.get(f"non_empty_patch_{cond}", "") for r in rows]
        empty_values = [r.get(f"empty_patch_{cond}", "") for r in rows]
        non_empty_count = sum(int(v) for v in non_empty_values if v in {"0", "1"})
        empty_count = sum(int(v) for v in empty_values if v in {"0", "1"})
        known_attempts = sum(1 for v in non_empty_values if v in {"0", "1"})
        status_error_count = 0
        patch_failed_count = sum(int(r.get(f"patch_failed_{cond}", "") or 0) for r in rows if r.get(f"patch_failed_{cond}", "") in {"0", "1"})
        no_patch_count = sum(int(r.get(f"no_patch_eval_{cond}", "") or 0) for r in rows if r.get(f"no_patch_eval_{cond}", "") in {"0", "1"})
        for r in rows:
            err = str(r.get(f"eval_error_{cond}", "")).strip()
            status = str(r.get(f"eval_status_{cond}", "")).lower()
            if err and "patch" not in status + " " + err and "no patch" not in status + " " + err:
                status_error_count += 1
        values_by_metric = {
            "wall_seconds": numeric_values(rows, f"wall_seconds_{cond}"),
            "llm_calls": numeric_values(rows, f"llm_calls_{cond}"),
            "input_tokens": numeric_values(rows, f"input_tokens_{cond}"),
            "output_tokens": numeric_values(rows, f"output_tokens_{cond}"),
            "total_tokens": numeric_values(rows, f"total_tokens_{cond}"),
            "max_turn_tokens": numeric_values(rows, f"max_turn_tokens_{cond}"),
            "patch_bytes": numeric_values(rows, f"patch_bytes_{cond}"),
            "patch_files_changed": numeric_values(rows, f"patch_files_changed_{cond}"),
        }
        condition_rows.append(
            {
                "condition": cond,
                "n_targets": n,
                "resolved_count": resolved_count if resolved_complete else "",
                "success_rate": f"{resolved_count / n:.6g}" if resolved_complete and n else "",
                "wilson_95_low": f"{low:.6g}" if resolved_complete else "",
                "wilson_95_high": f"{high:.6g}" if resolved_complete else "",
                "non_empty_patch_count": non_empty_count if known_attempts else "",
                "empty_patch_count": empty_count if known_attempts else "",
                "patch_attempt_rate": f"{non_empty_count / known_attempts:.6g}" if known_attempts else "",
                "eval_no_patch_count": no_patch_count if no_patch_count else "",
                "eval_patch_failed_count": patch_failed_count if patch_failed_count else "",
                "other_eval_error_count": status_error_count if status_error_count else "",
                "mean_wall_seconds": mean(values_by_metric["wall_seconds"]),
                "median_wall_seconds": median(values_by_metric["wall_seconds"]),
                "total_wall_seconds": total(values_by_metric["wall_seconds"]),
                "mean_llm_calls": mean(values_by_metric["llm_calls"]),
                "median_llm_calls": median(values_by_metric["llm_calls"]),
                "total_llm_calls": total(values_by_metric["llm_calls"]),
                "mean_input_tokens": mean(values_by_metric["input_tokens"]),
                "median_input_tokens": median(values_by_metric["input_tokens"]),
                "total_input_tokens": total(values_by_metric["input_tokens"]),
                "mean_output_tokens": mean(values_by_metric["output_tokens"]),
                "median_output_tokens": median(values_by_metric["output_tokens"]),
                "total_output_tokens": total(values_by_metric["output_tokens"]),
                "mean_total_tokens": mean(values_by_metric["total_tokens"]),
                "median_total_tokens": median(values_by_metric["total_tokens"]),
                "total_total_tokens": total(values_by_metric["total_tokens"]),
                "median_max_turn_tokens": median(values_by_metric["max_turn_tokens"]),
                "max_max_turn_tokens": f"{max(values_by_metric['max_turn_tokens']):.6g}" if values_by_metric["max_turn_tokens"] else "",
                "mean_patch_bytes": mean(values_by_metric["patch_bytes"]),
                "median_patch_bytes": median(values_by_metric["patch_bytes"]),
                "mean_patch_files_changed": mean(values_by_metric["patch_files_changed"]),
                "median_patch_files_changed": median(values_by_metric["patch_files_changed"]),
            }
        )
        resolved_attempts = 0
        unresolved_attempts = 0
        for r in rows:
            attempt = r.get(f"non_empty_patch_{cond}", "")
            res = r.get(f"resolved_{cond}", "")
            if attempt == "1" and res == "1":
                resolved_attempts += 1
            elif attempt == "1" and res == "0":
                unresolved_attempts += 1
        patch_quality.append(
            {
                "condition": cond,
                "resolved_count": resolved_count if resolved_complete else "",
                "non_empty_patch_count": non_empty_count if known_attempts else "",
                "resolved_given_non_empty_patch_rate": f"{resolved_attempts / non_empty_count:.6g}" if non_empty_count and resolved_complete else "",
                "empty_patch_count": empty_count if known_attempts else "",
                "unresolved_non_empty_patch_count": unresolved_attempts if resolved_complete and known_attempts else "",
            }
        )
    return condition_rows, patch_quality


def write_discovery_report(
    out_dir: Path,
    repo_root: Path,
    render_report: dict[str, Any],
    local_files: list[tuple[str, int]],
    remote_lines: list[str],
    remote_status: str,
    manual_remote_note: str,
) -> None:
    lines: list[str] = []
    lines.append("# ContextBench Posthoc Run Discovery\n")
    lines.append(f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    lines.append(f"- Local repo root: `{repo_root}`")
    lines.append(f"- Expected remote execution suffix: `{EXPECTED_REMOTE_EXECUTION_SUFFIX}`")
    lines.append("")
    lines.append("## Prompt Render Report")
    if render_report:
        for key in [
            "prompt_count",
            "unique_targets",
            "condition_counts",
            "forbidden_hit_count",
            "inconsistent_target_issue_hashes",
            "too_long_count",
            "prompt_manifest",
            "forbidden_scan",
        ]:
            if key in render_report:
                lines.append(f"- `{key}`: `{render_report[key]}`")
    else:
        lines.append("- Prompt render report was not found.")
    lines.append("")
    lines.append("## Remote Discovery")
    if manual_remote_note:
        lines.append(f"- Manual session note: {manual_remote_note}")
    lines.append(f"- Script status: {remote_status}")
    if remote_lines:
        lines.append("")
        lines.append("```text")
        lines.extend(remote_lines[:2000])
        if len(remote_lines) > 2000:
            lines.append(f"... truncated {len(remote_lines) - 2000} additional remote lines")
        lines.append("```")
    else:
        lines.append("- No remote paths were discovered by the script in this run.")
    lines.append("")
    lines.append("## Local Relevant Files")
    lines.append(f"- Relevant local file count: {len(local_files)}")
    lines.append("")
    lines.append("| path | bytes |")
    lines.append("|---|---:|")
    for path, size in local_files[:3000]:
        rel = path
        try:
            rel = str(Path(path).resolve().relative_to(repo_root.resolve()))
        except Exception:
            pass
        lines.append(f"| `{rel}` | {size} |")
    if len(local_files) > 3000:
        lines.append(f"| ... | truncated {len(local_files) - 3000} additional local paths |")
    write_text(out_dir / "run_discovery.md", "\n".join(lines) + "\n")


def prompt_parity_audit(prompt_rows: list[dict[str, Any]], render_report: dict[str, Any]) -> dict[str, Any]:
    hashes_by_target: dict[str, set[str]] = defaultdict(set)
    cond_counts = Counter()
    prompt_chars: dict[str, list[int]] = defaultdict(list)
    prior_chars: dict[str, list[int]] = defaultdict(list)
    for row in prompt_rows:
        target = str(row.get("target_instance_id", ""))
        cond = str(row.get("condition", ""))
        cond_counts[cond] += 1
        h = str(row.get("target_problem_statement_sha256", ""))
        if target and h:
            hashes_by_target[target].add(h)
        try:
            prompt_chars[cond].append(int(row.get("prompt_chars", 0)))
        except Exception:
            pass
        try:
            prior_chars[cond].append(int(row.get("prior_context_chars", 0)))
        except Exception:
            pass
    inconsistent = {k: sorted(v) for k, v in hashes_by_target.items() if len(v) > 1}
    return {
        "prompt_count": len(prompt_rows),
        "unique_targets": len(hashes_by_target),
        "condition_counts": dict(cond_counts),
        "inconsistent_target_issue_hashes": inconsistent,
        "forbidden_hit_count": render_report.get("forbidden_hit_count", ""),
        "forbidden_scan": render_report.get("forbidden_scan", ""),
        "prompt_chars": {k: stats(v) for k, v in prompt_chars.items()},
        "prior_context_chars": {k: stats(v) for k, v in prior_chars.items()},
        "only_prior_context_differs_supported": (not inconsistent and render_report.get("forbidden_hit_count", 0) == 0),
    }


def stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.fmean(values), 2),
        "max": max(values),
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")).replace("\n", " ") for c in columns) + " |")
    return "\n".join(out)


def write_reports(
    out_dir: Path,
    paired_rows: list[dict[str, Any]],
    validation_notes: list[str],
    parity: dict[str, Any],
    condition_rows: list[dict[str, Any]],
    patch_quality_rows: list[dict[str, Any]],
    missing_sources: list[str],
) -> None:
    counts_text = []
    for cond in CONDITIONS:
        values = [r.get(f"resolved_{cond}", "") for r in paired_rows]
        if all(v in {"0", "1"} for v in values):
            counts_text.append(f"- `{cond}`: {sum(int(v) for v in values)} / {len(values)}")
        else:
            counts_text.append(f"- `{cond}`: unavailable; {sum(1 for v in values if v not in {'0','1'})} missing evaluator values")

    analysis_lines = [
        "# ContextBench Posthoc Analysis Summary",
        "",
        "## Data Found",
        f"- Prompt manifest rows: {parity.get('prompt_count')}",
        f"- Unique prompt targets: {parity.get('unique_targets')}",
        f"- Prompt condition counts: `{parity.get('condition_counts')}`",
        f"- Forbidden prompt hit count: `{parity.get('forbidden_hit_count')}`",
        f"- Inconsistent target issue hashes across conditions: `{parity.get('inconsistent_target_issue_hashes')}`",
        "",
        "## Data Missing / Blocked",
    ]
    if missing_sources:
        analysis_lines.extend(f"- {item}" for item in missing_sources)
    else:
        analysis_lines.append("- No missing data was detected by collection.")
    analysis_lines.extend(
        [
            "",
            "## Main Aggregate Results",
            "Evaluator-backed aggregate counts from the paired table:",
            *counts_text,
            "",
            "Expected aggregate counts supplied in the request:",
            "- `no_memory`: 10 / 95",
            "- `raw`: 19 / 95",
            "- `adp`: 15 / 95",
            "- `memory`: 16 / 95",
            "",
            "## Validation",
        ]
    )
    if validation_notes:
        analysis_lines.extend(f"- {note}" for note in validation_notes)
    else:
        analysis_lines.append("- Paired table validation passed.")
    analysis_lines.extend(
        [
            "",
            "## Prompt / Context Audit",
            f"- Prompt count: {parity.get('prompt_count')}",
            f"- Valid target count expected after exclusion: 95",
            f"- Excluded target: `{EXCLUDED_TARGET}`",
            f"- Prompt character lengths by condition: `{parity.get('prompt_chars')}`",
            f"- Prior-context character lengths by condition: `{parity.get('prior_context_chars')}`",
            f"- Forbidden scan: `{parity.get('forbidden_scan')}` with hit count `{parity.get('forbidden_hit_count')}`",
            f"- Target issue text hashes identical across conditions: `{not bool(parity.get('inconsistent_target_issue_hashes'))}`",
            f"- Evidence that only PRIOR_CONTEXT differs: `{parity.get('only_prior_context_differs_supported')}` from prompt report/hash audit.",
            "",
            "## Recommended Paper Framing",
            (
                "Officially related prior SWE-agent experience improves downstream OpenHands/Qwen3.6 "
                "performance over no prior context. Raw trajectory context achieved the highest observed "
                "solve count, but differences among raw, ADP, and deterministic memory require paired "
                "statistical interpretation and should not be treated as universally established. The "
                "mechanism appears to involve localization/procedural transfer to the extent supported by "
                "overlap and transcript evidence."
            ),
            "",
            "## Status",
            "Paired statistical, overlap, transcript, and qualitative sections are populated by the companion scripts when evaluator/run artifacts are present.",
        ]
    )
    write_text(out_dir / "reports/analysis_summary.md", "\n".join(analysis_lines) + "\n")

    missing_lines = [
        "# Missing Data",
        "",
        "This file records fields that could not be derived from available artifacts.",
        "",
    ]
    if missing_sources:
        missing_lines.extend(f"- {item}" for item in missing_sources)
    else:
        missing_lines.append("- No missing sources detected.")
    missing_lines.extend(["", "## Validation Notes"])
    missing_lines.extend(f"- {note}" for note in validation_notes)
    write_text(out_dir / "missing_data.md", "\n".join(missing_lines) + "\n")

    readme = [
        "# ContextBench Posthoc Analysis Workspace",
        "",
        "Generated analysis workspace for the ContextBench prior-trajectory memory experiment.",
        "",
        "## Contents",
        "- `run_discovery.md`: local/remote path discovery notes.",
        "- `missing_data.md`: missing or blocked artifact sources.",
        "- `scripts/`: rerunnable collection/statistics/overlap/transcript scripts.",
        "- `data/`: CSV outputs.",
        "- `reports/`: Markdown summaries.",
        "- `figures/`: PNG figures when plotting inputs are available.",
        "",
        "## Rerun",
        "Run from the repository root:",
        "",
        "```bash",
        f"python {out_dir / 'scripts/collect_contextbench_posthoc.py'} --out-dir {out_dir}",
        f"python {out_dir / 'scripts/stats_contextbench_posthoc.py'} --out-dir {out_dir}",
        f"python {out_dir / 'scripts/overlap_contextbench_posthoc.py'} --out-dir {out_dir}",
        f"python {out_dir / 'scripts/transcript_mining_contextbench_posthoc.py'} --out-dir {out_dir}",
        "```",
        "",
        "For remote discovery, set `SSHPASS` in the shell and pass `--remote`; do not place the password in scripts or reports.",
    ]
    write_text(out_dir / "README.md", "\n".join(readme) + "\n")

    paper_tables = [
        "# Paper-Ready Tables",
        "",
        "## Condition Summary",
        markdown_table(condition_rows, ["condition", "n_targets", "resolved_count", "success_rate", "non_empty_patch_count", "patch_attempt_rate"]),
        "",
        "## Patch Quality Conditional",
        markdown_table(patch_quality_rows, PATCH_QUALITY_COLUMNS),
        "",
        "Additional tables are appended by the statistics, overlap, and transcript scripts when their source data is available.",
    ]
    write_text(out_dir / "reports/paper_ready_tables.md", "\n".join(paper_tables) + "\n")


def write_status_tables(out_dir: Path, paired_rows: list[dict[str, Any]]) -> None:
    write_csv(out_dir / "data/evaluator_status_by_target.csv", paired_rows, EVAL_STATUS_COLUMNS)
    write_csv(out_dir / "data/run_status_by_target.csv", paired_rows, RUN_STATUS_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", action="append", type=Path, default=[])
    parser.add_argument("--eval-root", action="append", type=Path, default=[])
    parser.add_argument("--remote", action="store_true", help="Attempt read-only remote discovery using SSHPASS/sshpass.")
    parser.add_argument("--manual-remote-note", default="")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    eprint(f"[collect] repo_root={repo_root}")
    eprint(f"[collect] out_dir={out_dir}")

    prompt_rows, pair_rows, run_rows, render_report = load_manifests(repo_root)
    parity = prompt_parity_audit(prompt_rows, render_report)
    eprint(f"[collect] loaded {len(prompt_rows)} prompt rows, {len(pair_rows)} pairs, {len(run_rows)} run rows")

    local_roots = [
        repo_root / "data/contextbench_phase2",
        repo_root / "reports",
        repo_root / "runs/contextbench_qwen3_30b_openhands",
    ]
    local_files: list[tuple[str, int]] = []
    for root in local_roots:
        local_files.extend(list_relevant_files(root))
    remote_lines: list[str] = []
    remote_status = "remote discovery not requested; pass --remote with SSHPASS set to enable."
    if args.remote:
        remote_lines, remote_status = ssh_remote_discovery()
    write_discovery_report(out_dir, repo_root, render_report, local_files, remote_lines, remote_status, args.manual_remote_note)

    eval_roots = [p.resolve() for p in args.eval_root if p.exists()]
    artifact_roots = [p.resolve() for p in args.artifact_root if p.exists()]
    eval_records = discover_eval_records(repo_root, eval_roots)
    eprint(f"[collect] parsed {len(eval_records)} evaluator records from local/eval roots")
    paired_rows, validation_notes = build_paired_rows(
        repo_root, prompt_rows, pair_rows, run_rows, artifact_roots, eval_records
    )
    write_csv(out_dir / "data/paired_results.csv", paired_rows, PAIRED_COLUMNS)
    condition_rows, patch_quality_rows = summarize_conditions(paired_rows)
    write_csv(out_dir / "data/condition_summary.csv", condition_rows, CONDITION_SUMMARY_COLUMNS)
    write_csv(out_dir / "data/patch_quality_conditional.csv", patch_quality_rows, PATCH_QUALITY_COLUMNS)
    write_status_tables(out_dir, paired_rows)

    missing_sources = []
    if not eval_records:
        missing_sources.append(
            "Evaluator result files for qwen36_*_95_keepimg were not found locally; remote SSH authentication is required to build resolved_* fields."
        )
    if not artifact_roots:
        missing_sources.append(
            "Full execution artifact root was not found/provided; run metadata, token accounting, patches, and transcripts are mostly unavailable."
        )
    if not remote_lines:
        missing_sources.append(
            f"Remote paths under {REMOTE_ROOTS} were not discovered by the script in this run."
        )
    write_reports(out_dir, paired_rows, validation_notes, parity, condition_rows, patch_quality_rows, missing_sources)

    eprint("[collect] wrote paired_results.csv, condition summaries, discovery, and missing-data reports")
    if validation_notes:
        eprint("[collect] validation did not pass; see missing_data.md and reports/analysis_summary.md")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
