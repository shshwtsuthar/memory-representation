#!/usr/bin/env python3
"""Transcript/behavior mining for ContextBench posthoc analysis."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CONDITIONS = ["no_memory", "raw", "adp", "memory"]
CASE_TYPES = [
    "raw_solved_memory_failed",
    "memory_solved_raw_failed",
    "raw_solved_adp_failed",
    "adp_solved_raw_failed",
    "all_prior_solved_no_memory_failed",
    "no_memory_solved_all_prior_failed",
    "high_overlap_prior_helped",
    "low_or_no_overlap_prior_helped",
    "high_overlap_prior_did_not_help",
    "empty_patch_no_memory_prior_non_empty_or_resolved",
]


TRANSCRIPT_COLUMNS = [
    "instance_id",
    "condition",
    "resolved",
    "non_empty_patch",
    "empty_patch",
    "run_status",
    "wall_seconds",
    "llm_calls",
    "iterations",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "max_turn_tokens",
    "num_bash_commands",
    "num_grep_commands",
    "num_find_commands",
    "num_rg_commands",
    "num_pytest_commands",
    "num_test_commands",
    "num_file_reads",
    "num_file_writes",
    "num_file_edits",
    "num_apply_patch_attempts",
    "num_git_diff_commands",
    "num_git_status_commands",
    "first_file_read",
    "first_file_edited",
    "all_files_read",
    "all_files_edited",
    "all_commands_run",
    "mentions_prior_context_bool",
    "mentions_prior_file_paths_bool",
    "mentions_prior_failure_signature_bool",
    "ran_tests_bool",
    "tests_passed_observed_bool",
    "tests_failed_observed_bool",
    "last_test_failure_signature",
    "last_runtime_error_signature",
    "timeout_bool",
    "tool_error_count",
    "model_empty_message_count",
    "security_risk_unknown_count",
]


COMMAND_USAGE_COLUMNS = [
    "condition",
    "n_runs",
    "runs_with_artifacts",
    "mean_bash_commands",
    "median_bash_commands",
    "mean_grep_commands",
    "mean_find_commands",
    "mean_rg_commands",
    "mean_pytest_commands",
    "mean_test_commands",
    "ran_tests_count",
    "tests_failed_observed_count",
    "tool_error_total",
    "timeout_count",
]


FILE_ACTIVITY_COLUMNS = [
    "condition",
    "n_runs",
    "runs_with_artifacts",
    "mean_file_reads",
    "median_file_reads",
    "mean_file_writes",
    "mean_file_edits",
    "mean_files_read_unique",
    "mean_files_edited_unique",
    "top_first_file_read",
    "top_first_file_edited",
]


FAILURE_COLUMNS = [
    "instance_id",
    "condition",
    "resolved",
    "tests_failed_observed_bool",
    "last_test_failure_signature",
    "last_runtime_error_signature",
    "timeout_bool",
    "tool_error_count",
    "security_risk_unknown_count",
]


QUAL_COLUMNS = [
    "case_type",
    "instance_id",
    "repo",
    "prior_instance_id",
    "target_instance_id",
    "condition_outcomes",
    "overlap_bucket",
    "localization_bucket",
    "hypothesized_mechanism",
    "supporting_artifact_paths",
    "notes",
]


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_path(path: str) -> str:
    path = path.replace("\\", "/").strip().strip("'\"")
    path = re.sub(r"^\./", "", path)
    for marker in ["/testbed/", "testbed/"]:
        if marker in path:
            path = path.split(marker, 1)[1]
    path = re.sub(r"^swebench_[^/]+/", "", path)
    parts = [p for p in path.split("/") if p not in {"", "."}]
    if parts and "__" in parts[0]:
        parts = parts[1:]
    return "/".join(parts)


def split_set(value: str) -> set[str]:
    if not value:
        return set()
    return {normalize_path(v) for v in value.split(";") if normalize_path(v)}


def set_str(values: Iterable[str], cap: int = 80) -> str:
    vals = sorted(v for v in set(values) if v)
    if len(vals) > cap:
        vals = vals[:cap] + [f"... truncated {len(vals) - cap} more"]
    return ";".join(vals)


def candidate_run_dirs(repo_root: Path, target: str, condition: str, artifact_roots: list[Path]) -> list[Path]:
    out = []
    for root in artifact_roots:
        out.extend([root / "runs" / target / condition, root / target / condition])
    out.extend(
        [
            repo_root / "data/contextbench_phase2/execution_full_qwen36_65k_fix1/runs" / target / condition,
            repo_root / "data/contextbench_phase2/execution/runs" / target / condition,
            repo_root / "runs/contextbench_qwen3_30b_openhands" / target / condition,
        ]
    )
    artifact_names = {"stdout.jsonl", "stderr.log", "run_meta.json", "patch.diff", "prediction.json", "prediction_audit.json"}
    return [p for p in out if p.exists() and any((p / name).exists() for name in artifact_names)]


def parse_patch_files(path: Path) -> set[str]:
    files: set[str] = set()
    if not path.exists():
        return files
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("diff --git "):
                parts = line.split()
                if len(parts) >= 4:
                    p = parts[3]
                    if p.startswith("b/"):
                        p = p[2:]
                    files.add(normalize_path(p))
    except OSError:
        pass
    return files


COMMAND_RE = re.compile(
    r"(?:\"command\"\s*:\s*\"(?P<jsoncmd>(?:\\.|[^\"])*)\"|(?:bash|python|pytest|git|grep|rg|find|sed|cat|ls)\s+[^\\n\"`]{0,400})",
    re.I,
)
PATH_RE = re.compile(r"(?:(?:[\w.-]+/)+[\w.@%+=:,~^# -]+\.(?:py|pyx|pyi|js|ts|tsx|jsx|java|c|cc|cpp|h|hpp|rs|go|rb|php|sh|toml|yaml|yml|ini|cfg|txt|rst|md|json|xml|html|css))")
FAIL_RE = re.compile(r"(FAILED [^\n]{1,220}|ERROR [^\n]{1,220}|Traceback \(most recent call last\):|AssertionError[^\n]{0,180}|Verifier setup failed[^\n]{0,220})")
RUNTIME_RE = re.compile(r"(Traceback \(most recent call last\):|Exception[^\n]{0,220}|RuntimeError[^\n]{0,220}|AttributeError[^\n]{0,220}|TypeError[^\n]{0,220}|ValueError[^\n]{0,220})")


def unescape_json_command(cmd: str) -> str:
    try:
        return json.loads(f'"{cmd}"')
    except Exception:
        return cmd


def scan_stdout(path: Path, prior_files: set[str]) -> dict[str, Any]:
    features: dict[str, Any] = {
        "num_bash_commands": 0,
        "num_grep_commands": 0,
        "num_find_commands": 0,
        "num_rg_commands": 0,
        "num_pytest_commands": 0,
        "num_test_commands": 0,
        "num_file_reads": 0,
        "num_file_writes": 0,
        "num_file_edits": 0,
        "num_apply_patch_attempts": 0,
        "num_git_diff_commands": 0,
        "num_git_status_commands": 0,
        "first_file_read": "",
        "first_file_edited": "",
        "all_files_read": "",
        "all_files_edited": "",
        "all_commands_run": "",
        "mentions_prior_context_bool": "0",
        "mentions_prior_file_paths_bool": "0",
        "mentions_prior_failure_signature_bool": "0",
        "ran_tests_bool": "0",
        "tests_passed_observed_bool": "0",
        "tests_failed_observed_bool": "0",
        "last_test_failure_signature": "",
        "last_runtime_error_signature": "",
        "timeout_bool": "0",
        "tool_error_count": 0,
        "model_empty_message_count": 0,
        "security_risk_unknown_count": 0,
    }
    commands: list[str] = []
    files_read: list[str] = []
    files_edited: list[str] = []
    if not path.exists():
        return features
    prior_file_sample = {p for p in prior_files if p}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.rstrip("\n")
            low = text.lower()
            if "prior context" in low or "prior issue" in low or "prior trajectory" in low:
                features["mentions_prior_context_bool"] = "1"
            if "failure signature" in low or "traceback" in low and "prior" in low:
                features["mentions_prior_failure_signature_bool"] = "1"
            if "timeout" in low or "timed out" in low:
                features["timeout_bool"] = "1"
            if "tool error" in low or "\"error\"" in low or "tool_error" in low:
                features["tool_error_count"] += 1
            if "security risk unknown" in low:
                features["security_risk_unknown_count"] += 1
            if "empty message" in low or "empty response" in low:
                features["model_empty_message_count"] += 1
            if "passed" in low and ("pytest" in low or "test" in low):
                features["tests_passed_observed_bool"] = "1"
            if "failed" in low and ("pytest" in low or "test" in low):
                features["tests_failed_observed_bool"] = "1"
            for m in FAIL_RE.finditer(text):
                features["last_test_failure_signature"] = m.group(1)[:240]
                features["tests_failed_observed_bool"] = "1"
            for m in RUNTIME_RE.finditer(text):
                features["last_runtime_error_signature"] = m.group(1)[:240]
            # Tool-function hints.
            if re.search(r'\b(Read|open_file|str_replace_editor)\b', text):
                paths = [normalize_path(p) for p in PATH_RE.findall(text)]
                for p in paths:
                    files_read.append(p)
                    if not features["first_file_read"]:
                        features["first_file_read"] = p
                if paths:
                    features["num_file_reads"] += len(paths)
            if re.search(r'\b(Write|Edit|str_replace|insert|apply_patch)\b', text):
                paths = [normalize_path(p) for p in PATH_RE.findall(text)]
                for p in paths:
                    files_edited.append(p)
                    if not features["first_file_edited"]:
                        features["first_file_edited"] = p
                if paths:
                    features["num_file_edits"] += len(paths)
            if "apply_patch" in low:
                features["num_apply_patch_attempts"] += 1
            for m in COMMAND_RE.finditer(text):
                cmd = m.group("jsoncmd")
                if cmd:
                    cmd = unescape_json_command(cmd)
                else:
                    cmd = m.group(0)
                cmd = re.sub(r"\s+", " ", cmd).strip()
                if not cmd:
                    continue
                commands.append(cmd[:500])
                clow = cmd.lower()
                features["num_bash_commands"] += 1
                if "grep" in clow:
                    features["num_grep_commands"] += 1
                if re.search(r"(^|\s)find\s", clow):
                    features["num_find_commands"] += 1
                if re.search(r"(^|\s)rg\s", clow):
                    features["num_rg_commands"] += 1
                if "pytest" in clow:
                    features["num_pytest_commands"] += 1
                if "pytest" in clow or re.search(r"\btest\b", clow):
                    features["num_test_commands"] += 1
                    features["ran_tests_bool"] = "1"
                if "git diff" in clow:
                    features["num_git_diff_commands"] += 1
                if "git status" in clow:
                    features["num_git_status_commands"] += 1
                if re.search(r"(^|\s)(cat|sed|head|tail|less|nl)\s", clow):
                    paths = [normalize_path(p) for p in PATH_RE.findall(cmd)]
                    for p in paths:
                        files_read.append(p)
                        if not features["first_file_read"]:
                            features["first_file_read"] = p
                    features["num_file_reads"] += len(paths)
    edited_set = set(files_edited)
    read_set = set(files_read)
    if prior_file_sample and (prior_file_sample & read_set or prior_file_sample & edited_set):
        features["mentions_prior_file_paths_bool"] = "1"
    features["all_files_read"] = set_str(read_set)
    features["all_files_edited"] = set_str(edited_set)
    features["all_commands_run"] = ";".join(commands[:100])
    if len(commands) > 100:
        features["all_commands_run"] += f";... truncated {len(commands) - 100} more"
    return features


def mine_run(
    repo_root: Path,
    paired: dict[str, str],
    condition: str,
    prior_files: set[str],
    artifact_roots: list[Path],
) -> dict[str, Any]:
    target = paired["instance_id"]
    base = {
        "instance_id": target,
        "condition": condition,
        "resolved": paired.get(f"resolved_{condition}", ""),
        "non_empty_patch": paired.get(f"non_empty_patch_{condition}", ""),
        "empty_patch": paired.get(f"empty_patch_{condition}", ""),
        "run_status": paired.get(f"run_status_{condition}", ""),
        "wall_seconds": paired.get(f"wall_seconds_{condition}", ""),
        "llm_calls": paired.get(f"llm_calls_{condition}", ""),
        "iterations": paired.get(f"openhands_iterations_{condition}", ""),
        "input_tokens": paired.get(f"input_tokens_{condition}", ""),
        "output_tokens": paired.get(f"output_tokens_{condition}", ""),
        "total_tokens": paired.get(f"total_tokens_{condition}", ""),
        "max_turn_tokens": paired.get(f"max_turn_tokens_{condition}", ""),
    }
    for run_dir in candidate_run_dirs(repo_root, target, condition, artifact_roots):
        features = scan_stdout(run_dir / "stdout.jsonl", prior_files)
        edited_from_patch = parse_patch_files(run_dir / "patch.diff")
        if edited_from_patch:
            existing = split_set(str(features.get("all_files_edited", "")))
            features["all_files_edited"] = set_str(existing | edited_from_patch)
            features["num_file_edits"] = max(int(features.get("num_file_edits", 0)), len(edited_from_patch))
            if not features.get("first_file_edited"):
                features["first_file_edited"] = sorted(edited_from_patch)[0]
        base.update(features)
        return base
    # Fill absent feature columns.
    for col in TRANSCRIPT_COLUMNS:
        base.setdefault(col, "")
    for col in [
        "num_bash_commands",
        "num_grep_commands",
        "num_find_commands",
        "num_rg_commands",
        "num_pytest_commands",
        "num_test_commands",
        "num_file_reads",
        "num_file_writes",
        "num_file_edits",
        "num_apply_patch_attempts",
        "num_git_diff_commands",
        "num_git_status_commands",
        "tool_error_count",
        "model_empty_message_count",
        "security_risk_unknown_count",
    ]:
        base[col] = ""
    return base


def mean(vals: list[float]) -> str:
    return "" if not vals else f"{statistics.fmean(vals):.6g}"


def median(vals: list[float]) -> str:
    return "" if not vals else f"{statistics.median(vals):.6g}"


def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(key, "")))
        except (TypeError, ValueError):
            pass
    return vals


def aggregate_commands(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cond in CONDITIONS:
        group = [r for r in rows if r["condition"] == cond]
        artifact_group = [r for r in group if r.get("num_bash_commands") != ""]
        out.append(
            {
                "condition": cond,
                "n_runs": len(group),
                "runs_with_artifacts": len(artifact_group),
                "mean_bash_commands": mean(numeric(artifact_group, "num_bash_commands")),
                "median_bash_commands": median(numeric(artifact_group, "num_bash_commands")),
                "mean_grep_commands": mean(numeric(artifact_group, "num_grep_commands")),
                "mean_find_commands": mean(numeric(artifact_group, "num_find_commands")),
                "mean_rg_commands": mean(numeric(artifact_group, "num_rg_commands")),
                "mean_pytest_commands": mean(numeric(artifact_group, "num_pytest_commands")),
                "mean_test_commands": mean(numeric(artifact_group, "num_test_commands")),
                "ran_tests_count": sum(1 for r in artifact_group if r.get("ran_tests_bool") == "1"),
                "tests_failed_observed_count": sum(1 for r in artifact_group if r.get("tests_failed_observed_bool") == "1"),
                "tool_error_total": sum(int(float(r.get("tool_error_count", 0) or 0)) for r in artifact_group),
                "timeout_count": sum(1 for r in artifact_group if r.get("timeout_bool") == "1"),
            }
        )
    return out


def top_value(values: list[str]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        if v:
            counts[v] += 1
    if not counts:
        return ""
    k = max(counts, key=lambda x: counts[x])
    return f"{k} ({counts[k]})"


def aggregate_files(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cond in CONDITIONS:
        group = [r for r in rows if r["condition"] == cond]
        artifact_group = [r for r in group if r.get("num_file_reads") != ""]
        unique_reads = [len(split_set(str(r.get("all_files_read", "")))) for r in artifact_group]
        unique_edits = [len(split_set(str(r.get("all_files_edited", "")))) for r in artifact_group]
        out.append(
            {
                "condition": cond,
                "n_runs": len(group),
                "runs_with_artifacts": len(artifact_group),
                "mean_file_reads": mean(numeric(artifact_group, "num_file_reads")),
                "median_file_reads": median(numeric(artifact_group, "num_file_reads")),
                "mean_file_writes": mean(numeric(artifact_group, "num_file_writes")),
                "mean_file_edits": mean(numeric(artifact_group, "num_file_edits")),
                "mean_files_read_unique": mean([float(x) for x in unique_reads]),
                "mean_files_edited_unique": mean([float(x) for x in unique_edits]),
                "top_first_file_read": top_value([str(r.get("first_file_read", "")) for r in artifact_group]),
                "top_first_file_edited": top_value([str(r.get("first_file_edited", "")) for r in artifact_group]),
            }
        )
    return out


def qualitative_cases(paired_rows: list[dict[str, str]], overlap_rows: list[dict[str, str]], transcript_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(r.get(f"resolved_{c}", "") not in {"0", "1"} for r in paired_rows for c in CONDITIONS):
        return []
    overlap = {r["instance_id"]: r for r in overlap_rows}
    transcript_by_target_cond = {(r["instance_id"], r["condition"]): r for r in transcript_rows}

    def outcomes(row: dict[str, str]) -> dict[str, int]:
        return {c: int(row[f"resolved_{c}"]) for c in CONDITIONS}

    selectors = {
        "raw_solved_memory_failed": lambda o, r: o["raw"] == 1 and o["memory"] == 0,
        "memory_solved_raw_failed": lambda o, r: o["memory"] == 1 and o["raw"] == 0,
        "raw_solved_adp_failed": lambda o, r: o["raw"] == 1 and o["adp"] == 0,
        "adp_solved_raw_failed": lambda o, r: o["adp"] == 1 and o["raw"] == 0,
        "all_prior_solved_no_memory_failed": lambda o, r: o["no_memory"] == 0 and all(o[c] == 1 for c in ["raw", "adp", "memory"]),
        "no_memory_solved_all_prior_failed": lambda o, r: o["no_memory"] == 1 and all(o[c] == 0 for c in ["raw", "adp", "memory"]),
        "high_overlap_prior_helped": lambda o, r: r.get("overlap_bucket") == "same_file_overlap" and o["no_memory"] == 0 and any(o[c] == 1 for c in ["raw", "adp", "memory"]),
        "low_or_no_overlap_prior_helped": lambda o, r: r.get("overlap_bucket") in {"no_gold_file_overlap", "same_directory_only"} and o["no_memory"] == 0 and any(o[c] == 1 for c in ["raw", "adp", "memory"]),
        "high_overlap_prior_did_not_help": lambda o, r: r.get("overlap_bucket") == "same_file_overlap" and not any(o[c] == 1 for c in ["raw", "adp", "memory"]),
        "empty_patch_no_memory_prior_non_empty_or_resolved": lambda o, r: False,
    }
    selected: list[dict[str, Any]] = []
    for case_type in CASE_TYPES:
        for row in paired_rows:
            ov = overlap.get(row["instance_id"], {})
            o = outcomes(row)
            if case_type == "empty_patch_no_memory_prior_non_empty_or_resolved":
                cond = row.get("empty_patch_no_memory") == "1" and any(row.get(f"non_empty_patch_{c}") == "1" or row.get(f"resolved_{c}") == "1" for c in ["raw", "adp", "memory"])
            else:
                cond = selectors[case_type](o, ov)
            if not cond:
                continue
            artifact_paths = []
            for c in CONDITIONS:
                tr = transcript_by_target_cond.get((row["instance_id"], c), {})
                if tr.get("all_files_edited"):
                    artifact_paths.append(f"{c}: edited {tr.get('all_files_edited')}")
            mechanism = "unknown"
            if ov.get("overlap_bucket") == "same_file_overlap":
                mechanism = "same_file_transfer"
            elif ov.get("localization_bucket") == "prior_trajectory_same_directory_as_target_gold":
                mechanism = "same_directory_transfer"
            elif case_type.startswith("raw_"):
                mechanism = "raw_redundancy_helped"
            selected.append(
                {
                    "case_type": case_type,
                    "instance_id": row["instance_id"],
                    "repo": row.get("repo", ""),
                    "prior_instance_id": row.get("prior_instance_id", ""),
                    "target_instance_id": row.get("target_instance_id", row["instance_id"]),
                    "condition_outcomes": ";".join(f"{c}={o[c]}" for c in CONDITIONS),
                    "overlap_bucket": ov.get("overlap_bucket", ""),
                    "localization_bucket": ov.get("localization_bucket", ""),
                    "hypothesized_mechanism": mechanism,
                    "supporting_artifact_paths": " | ".join(artifact_paths)[:1000],
                    "notes": "Mechanism is a hypothesis pending manual transcript review.",
                }
            )
            break
    return selected


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")).replace("\n", " ") for c in columns) + " |")
    return "\n".join(out)


def write_reports(
    out_dir: Path,
    transcript_rows: list[dict[str, Any]],
    command_rows: list[dict[str, Any]],
    file_rows: list[dict[str, Any]],
    qual_rows: list[dict[str, Any]],
) -> None:
    artifacts = sum(1 for r in transcript_rows if r.get("num_bash_commands") != "")
    lines = [
        "# Transcript Behavior Results",
        "",
        f"- Target-condition rows: {len(transcript_rows)}",
        f"- Rows with run artifacts mined: {artifacts}",
        "",
        "## Command Usage",
        markdown_table(command_rows, COMMAND_USAGE_COLUMNS),
        "",
        "## File Activity",
        markdown_table(file_rows, FILE_ACTIVITY_COLUMNS),
    ]
    if artifacts == 0:
        lines.append("")
        lines.append("Full OpenHands stdout/stderr/conversation artifacts were not available locally, so behavior mining is schema-only in this run.")
    write_text(out_dir / "reports/transcript_behavior_results.md", "\n".join(lines) + "\n")

    qlines = ["# Qualitative Case Notes", ""]
    if not qual_rows:
        qlines.append("Qualitative cases could not be selected because paired evaluator outcomes are incomplete or overlap buckets are unknown.")
    else:
        for row in qual_rows:
            qlines.extend(
                [
                    f"## {row['case_type']}: {row['instance_id']}",
                    f"- repo: `{row.get('repo','')}`",
                    f"- prior_instance_id: `{row.get('prior_instance_id','')}`",
                    f"- condition_outcomes: `{row.get('condition_outcomes','')}`",
                    f"- overlap: `{row.get('overlap_bucket','')}` / `{row.get('localization_bucket','')}`",
                    f"- hypothesized_mechanism: `{row.get('hypothesized_mechanism','unknown')}`",
                    f"- supporting_artifact_paths: {row.get('supporting_artifact_paths','')}",
                    "- short capped excerpts: unavailable in automated summary; inspect artifacts listed above.",
                    "",
                ]
            )
    write_text(out_dir / "reports/qualitative_case_notes.md", "\n".join(qlines) + "\n")

    # Append runtime/token summary to paper-ready tables without assuming prior content.
    paper_path = out_dir / "reports/paper_ready_tables.md"
    existing = paper_path.read_text(encoding="utf-8") if paper_path.exists() else "# Paper-Ready Tables\n"
    additions: list[str] = []
    if "## Transcript Behavior Summary" not in existing:
        additions.extend(
            [
                "",
                "## Transcript Behavior Summary",
                markdown_table(command_rows, ["condition", "n_runs", "runs_with_artifacts", "mean_bash_commands", "ran_tests_count", "timeout_count"]),
            ]
        )
    if "## Qualitative Case Index" not in existing:
        additions.extend(
            [
                "",
                "## Qualitative Case Index",
                markdown_table(qual_rows, QUAL_COLUMNS) if qual_rows else "_Unavailable: missing paired outcomes and/or run artifacts._",
            ]
        )
    if additions:
        write_text(paper_path, existing.rstrip() + "\n" + "\n".join(additions) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", action="append", type=Path, default=[])
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    artifact_roots = [p.resolve() for p in args.artifact_root if p.exists()]

    paired_rows = read_csv(out_dir / "data/paired_results.csv")
    overlap_rows = read_csv(out_dir / "data/overlap_features.csv")
    overlap_by_target = {r["instance_id"]: r for r in overlap_rows}
    eprint(f"[transcript] loaded {len(paired_rows)} paired rows")
    transcript_rows: list[dict[str, Any]] = []
    for paired in paired_rows:
        prior_files = split_set(overlap_by_target.get(paired["instance_id"], {}).get("prior_trajectory_inspected_files", "")) | split_set(
            overlap_by_target.get(paired["instance_id"], {}).get("prior_trajectory_edited_files", "")
        )
        for cond in CONDITIONS:
            transcript_rows.append(mine_run(repo_root, paired, cond, prior_files, artifact_roots))
    write_csv(out_dir / "data/transcript_behavior_features.csv", transcript_rows, TRANSCRIPT_COLUMNS)
    command_rows = aggregate_commands(transcript_rows)
    file_rows = aggregate_files(transcript_rows)
    failure_rows = [
        {col: row.get(col, "") for col in FAILURE_COLUMNS}
        for row in transcript_rows
        if row.get("tests_failed_observed_bool") == "1" or row.get("last_runtime_error_signature") or row.get("timeout_bool") == "1"
    ]
    write_csv(out_dir / "data/command_usage_by_condition.csv", command_rows, COMMAND_USAGE_COLUMNS)
    write_csv(out_dir / "data/file_activity_by_condition.csv", file_rows, FILE_ACTIVITY_COLUMNS)
    write_csv(out_dir / "data/failure_signature_features.csv", failure_rows, FAILURE_COLUMNS)

    qual_rows = qualitative_cases(paired_rows, overlap_rows, transcript_rows)
    write_csv(out_dir / "data/qualitative_case_index.csv", qual_rows, QUAL_COLUMNS)
    write_reports(out_dir, transcript_rows, command_rows, file_rows, qual_rows)
    if not any(r.get("num_bash_commands") != "" for r in transcript_rows):
        eprint("[transcript] no full run artifacts found; wrote schema-correct transcript outputs")
        return 2
    eprint("[transcript] completed transcript mining")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
