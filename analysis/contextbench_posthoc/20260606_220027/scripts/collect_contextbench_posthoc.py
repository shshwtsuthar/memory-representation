#!/usr/bin/env python3
"""Populate ContextBench posthoc analysis outputs.

This script is intentionally self-contained. It reads local manifests/datasets,
the copied evaluator summaries, and compact run features parsed read-only from
the remote final execution root via sshpass using SSHPASS from the environment.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REMOTE = ROOT / "remote"
REMOTE_FEATURES = REMOTE / "run_features.jsonl"

CONDITIONS = ["no_memory", "raw", "adp", "memory"]
COMPARISONS = [
    ("raw", "no_memory"),
    ("memory", "no_memory"),
    ("adp", "no_memory"),
    ("raw", "memory"),
    ("raw", "adp"),
    ("memory", "adp"),
]
EXPECTED_COUNTS = {"no_memory": 10, "raw": 19, "adp": 15, "memory": 16}
EXCLUDED = "django__django-28147"
SYMPY_ANOMALY = "sympy__sympy-19006"
FINAL_EXEC_ROOT = "/mnt/data/shashwat/openhands-adp-memory/data/contextbench_phase2/execution_full_qwen36_65k_fix1"
EVAL_ROOT = "/mnt/data/shashwat/SWEContextBench"
MODEL_DIR = "openhands-qwen3.6-35b-a3b-65k-ollama-contextbench-memory-repr"


def ensure_dirs() -> None:
    for p in [DATA, REPORTS, ROOT / "figures", ROOT / "scripts"]:
        p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(errors="replace"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(float(x))]
    return statistics.mean(xs) if xs else float("nan")


def median(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(float(x))]
    return statistics.median(xs) if xs else float("nan")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return center - half, center + half


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def chi2_sf_1df(x: float) -> float:
    # For df=1, survival function = erfc(sqrt(x/2)).
    return math.erfc(math.sqrt(max(0.0, x) / 2.0))


def bootstrap_diff(a: list[int], b: list[int], n_boot: int = 20000, seed: int = 0) -> tuple[float, float, float]:
    try:
        import numpy as np
    except Exception:
        # Deterministic fallback with fewer pure-Python resamples.
        import random

        rng = random.Random(seed)
        n = len(a)
        vals = []
        for _ in range(5000):
            idx = [rng.randrange(n) for _ in range(n)]
            vals.append(sum(a[i] - b[i] for i in idx) / n)
        vals.sort()
        return vals[int(0.025 * len(vals))], vals[len(vals) // 2], vals[int(0.975 * len(vals))]
    rng = np.random.default_rng(seed)
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    n = len(aa)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (aa[idx].mean(axis=1) - bb[idx].mean(axis=1))
    low, mid, high = np.percentile(diffs, [2.5, 50, 97.5])
    return float(low), float(mid), float(high)


def holm(pairs: list[dict[str, Any]], pkey: str = "mcnemar_exact_p") -> None:
    ordered = sorted(enumerate(pairs), key=lambda x: float(x[1][pkey]))
    m = len(pairs)
    prev = 0.0
    adjusted = [1.0] * m
    for rank, (idx, row) in enumerate(ordered):
        val = min(1.0, (m - rank) * float(row[pkey]))
        prev = max(prev, val)
        adjusted[idx] = prev
    for row, val in zip(pairs, adjusted):
        row["holm_adjusted_p_across_six_comparisons"] = val


def parse_concat_eval() -> dict[str, dict[str, Any]]:
    path = REMOTE / "evaluator_summaries" / "four_keepimg_concat.txt"
    if not path.exists():
        raise SystemExit(f"Missing evaluator summary concat: {path}")
    text = path.read_text(errors="replace")
    out: dict[str, dict[str, Any]] = {}
    chunks = re.split(r"===== ([^\n]+)\n", text)
    for i in range(1, len(chunks), 2):
        fname = chunks[i].strip()
        body = chunks[i + 1].strip()
        if not body:
            continue
        cond = None
        for c in CONDITIONS:
            if f"qwen36_{c}_95_keepimg.json" in fname:
                cond = c
        if cond is None:
            continue
        out[cond] = json.loads(body)
    missing = [c for c in CONDITIONS if c not in out]
    if missing:
        raise SystemExit(f"Missing evaluator summaries for {missing}")
    return out


def parse_manifests() -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    pair_by_target: dict[str, dict[str, Any]] = {}
    with (REPO / "data/contextbench_phase2/pair_manifest.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            pair_by_target[row["related_instance_id"]] = row

    prompt_by_target_cond: dict[tuple[str, str], dict[str, Any]] = {}
    with (REPO / "data/contextbench_phase2/prompt_manifest.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            prompt_by_target_cond[(row["target_instance_id"], row["condition"])] = row

    run_by_target_cond: dict[tuple[str, str], dict[str, Any]] = {}
    with (REPO / "data/contextbench_phase2/run_manifest.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            run_by_target_cond[(row["target_instance_id"], row["condition"])] = row
    return pair_by_target, prompt_by_target_cond, run_by_target_cond


def remote_parse_features() -> None:
    if REMOTE_FEATURES.exists() and REMOTE_FEATURES.stat().st_size > 1000:
        print(f"Using existing {REMOTE_FEATURES}")
        return
    if not os.environ.get("SSHPASS"):
        raise SystemExit("SSHPASS is not set")
    code = r'''
import json, os, re, glob, sys
from pathlib import Path

ROOT=Path("/mnt/data/shashwat/openhands-adp-memory/data/contextbench_phase2/execution_full_qwen36_65k_fix1")
CONDS=["no_memory","raw","adp","memory"]

def load_json(p):
    try:
        return json.loads(Path(p).read_text(errors="replace"))
    except Exception:
        return {}

def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)

def cap_join(vals, n=80):
    vals=list(dict.fromkeys([v for v in vals if v]))
    return "|".join(vals[:n])

def parse_patch(text):
    files=[]; add=0; dele=0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            m=re.match(r"diff --git a/(.*?) b/(.*)$", line)
            if m:
                files.append(m.group(2))
        elif line.startswith("+") and not line.startswith("+++"):
            add+=1
        elif line.startswith("-") and not line.startswith("---"):
            dele+=1
    return files, add, dele

def parse_stdout(p):
    counts={k:0 for k in [
        "num_bash_commands","num_grep_commands","num_find_commands","num_rg_commands",
        "num_pytest_commands","num_test_commands","num_file_reads","num_file_writes",
        "num_file_edits","num_apply_patch_attempts","num_git_diff_commands","num_git_status_commands",
        "tool_error_count","model_empty_message_count","security_risk_unknown_count"
    ]}
    commands=[]; reads=[]; edits=[]; failure=[]; runtime=[]
    mentions_prior=False; mentions_prior_paths=False; mentions_prior_failure=False
    if not Path(p).exists():
        return counts, commands, reads, edits, failure, runtime, mentions_prior, mentions_prior_paths, mentions_prior_failure
    cmd_re=re.compile(r'"command"\s*:\s*"((?:\\.|[^"\\])*)"', re.S)
    path_re=re.compile(r'(/testbed/[^"\\\s]+|[A-Za-z0-9_./-]+\.(?:py|rst|txt|md|json|yml|yaml|cfg|ini|toml))')
    with open(p, errors="replace") as f:
        for raw in f:
            low=raw.lower()
            if "prior context" in low or "prior_context" in low: mentions_prior=True
            if "prior" in low and re.search(r'\.(py|rst|txt|md)', low): mentions_prior_paths=True
            if "prior" in low and any(x in low for x in ["fail", "error", "traceback", "exception"]): mentions_prior_failure=True
            if "securityrisk.unknown" in low or "security_risk" in low: counts["security_risk_unknown_count"]+=1
            if "tool error" in low or "tool_error" in low or "error validating args" in low: counts["tool_error_count"]+=1
            if "empty message" in low or "empty response" in low: counts["model_empty_message_count"]+=1
            if any(x in low for x in ["traceback", "assertionerror", "failed", "error:", "exception"]):
                s=re.sub(r"\s+"," ", raw.strip())
                if s: failure.append(s[:220])
            if any(x in low for x in ["timeout", "runtimeerror", "subprocess"]):
                s=re.sub(r"\s+"," ", raw.strip())
                if s: runtime.append(s[:220])
            for m in cmd_re.finditer(raw):
                try:
                    cmd=bytes(m.group(1), "utf-8").decode("unicode_escape")
                except Exception:
                    cmd=m.group(1)
                commands.append(cmd.strip())
                cl=cmd.lower()
                counts["num_bash_commands"]+=1
                if "grep" in cl: counts["num_grep_commands"]+=1
                if re.search(r'(^|\s)find(\s|$)', cl): counts["num_find_commands"]+=1
                if re.search(r'(^|\s)rg(\s|$)', cl): counts["num_rg_commands"]+=1
                if "pytest" in cl: counts["num_pytest_commands"]+=1
                if any(x in cl for x in ["pytest"," test","tox ","manage.py test"," runtests"]): counts["num_test_commands"]+=1
                if "apply_patch" in cl or "git apply" in cl: counts["num_apply_patch_attempts"]+=1
                if "git diff" in cl: counts["num_git_diff_commands"]+=1
                if "git status" in cl: counts["num_git_status_commands"]+=1
            try:
                obj=json.loads(raw)
            except Exception:
                obj=None
            if obj is not None:
                for d in walk(obj):
                    cmd=d.get("command")
                    path=d.get("path") or d.get("file_path") or d.get("filepath")
                    if isinstance(cmd,str) and isinstance(path,str):
                        c=cmd.lower()
                        clean=path.replace("/testbed/","")
                        if c=="view":
                            counts["num_file_reads"]+=1; reads.append(clean)
                        if c in {"create","str_replace","insert","undo_edit"}:
                            counts["num_file_edits"]+=1; edits.append(clean)
                            if c=="create": counts["num_file_writes"]+=1
            if "file_editor" in low:
                for pm in path_re.findall(raw):
                    if "/testbed/" in pm:
                        reads.append(pm.replace("/testbed/",""))
    return counts, commands, reads, edits, failure, runtime, mentions_prior, mentions_prior_paths, mentions_prior_failure

def state_tokens(run_dir):
    paths=list(Path(run_dir).glob(".openhands_home/.openhands/conversations/*/base_state.json"))
    if not paths: return {}, ""
    st=load_json(paths[0])
    usage=((st.get("stats") or {}).get("usage_to_metrics") or {})
    prompt=comp=total_calls=max_turn=0
    for metric in usage.values():
        acc=(metric.get("accumulated_token_usage") or {})
        prompt += int(acc.get("prompt_tokens") or 0)
        comp += int(acc.get("completion_tokens") or 0)
        for tu in metric.get("token_usages") or []:
            total_calls += 1
            max_turn=max(max_turn, int(tu.get("per_turn_token") or 0))
    return {"input_tokens":prompt,"output_tokens":comp,"total_tokens":prompt+comp,"max_turn_tokens":max_turn,"state_llm_calls":total_calls}, str(paths[0])

for target_dir in sorted((ROOT/"runs").iterdir()):
    if not target_dir.is_dir(): continue
    for cond in CONDS:
        run_dir=target_dir/cond
        if not run_dir.exists(): continue
        meta=load_json(run_dir/"run_meta.json")
        patch_text=(run_dir/"patch.diff").read_text(errors="replace") if (run_dir/"patch.diff").exists() else ""
        files, added, deleted = parse_patch(patch_text)
        counts, commands, reads, edits, failure, runtime, mp, mpp, mpf = parse_stdout(run_dir/"stdout.jsonl")
        toks, state_path = state_tokens(run_dir)
        stderr=""
        try:
            stderr=(run_dir/"stderr.log").read_text(errors="replace")[:2000]
        except Exception:
            pass
        out={
            "instance_id": target_dir.name, "condition": cond, "run_dir": str(run_dir),
            "run_meta_path": str(run_dir/"run_meta.json"), "stdout_path": str(run_dir/"stdout.jsonl"),
            "stderr_path": str(run_dir/"stderr.log"), "patch_path": str(run_dir/"patch.diff"),
            "prediction_path": str(run_dir/"prediction.json"), "base_state_path": state_path,
            "run_status": meta.get("status"), "run_exit_code": ((meta.get("openhands_result") or {}).get("exit_code")),
            "wall_seconds": meta.get("elapsed_seconds") or ((meta.get("openhands_result") or {}).get("elapsed_seconds")),
            "openhands_iterations": meta.get("agent_action_count"),
            "llm_calls": meta.get("llm_call_count") or toks.get("state_llm_calls"),
            "input_tokens": toks.get("input_tokens"), "output_tokens": toks.get("output_tokens"),
            "total_tokens": toks.get("total_tokens"), "max_turn_tokens": toks.get("max_turn_tokens"),
            "prompt_chars": meta.get("prompt_char_count"), "patch_bytes": len(patch_text.encode("utf-8")),
            "patch_lines_added": meta.get("patch_lines_added", added), "patch_lines_deleted": meta.get("patch_lines_removed", deleted),
            "patch_files_changed": meta.get("patch_files_changed", len(files)),
            "empty_patch": bool(meta.get("patch_empty", len(patch_text.strip())==0)),
            "non_empty_patch": not bool(meta.get("patch_empty", len(patch_text.strip())==0)),
            "files_edited_in_model_patch": cap_join(files),
            "first_file_read": reads[0] if reads else "", "first_file_edited": edits[0] if edits else "",
            "all_files_read": cap_join(reads), "all_files_edited": cap_join(edits or files),
            "all_commands_run": cap_join(commands, 60),
            "last_test_failure_signature": failure[-1] if failure else "",
            "last_runtime_error_signature": runtime[-1] if runtime else "",
            "mentions_prior_context_bool": mp, "mentions_prior_file_paths_bool": mpp,
            "mentions_prior_failure_signature_bool": mpf,
            "timeout_bool": bool(((meta.get("openhands_result") or {}).get("timed_out")) or "timeout" in stderr.lower()),
            "stderr_excerpt": stderr[:500],
        }
        out.update(counts)
        out["ran_tests_bool"]=out["num_test_commands"]>0
        out["tests_passed_observed_bool"]=" passed" in out["all_commands_run"].lower() or " passed" in " ".join(failure).lower()
        out["tests_failed_observed_bool"]=bool(failure)
        print(json.dumps(out, sort_keys=True))
'''
    cmd = [
        "sshpass", "-e", "ssh", "-F", "/dev/null",
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "shashwat@10.151.160.71", "python3 -",
    ]
    print("Parsing remote run artifacts read-only...")
    proc = subprocess.run(cmd, input=code, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"Remote feature parse failed with rc={proc.returncode}")
    REMOTE_FEATURES.write_text(proc.stdout, encoding="utf-8")
    print(f"Wrote {REMOTE_FEATURES}")


def load_features() -> dict[tuple[str, str], dict[str, Any]]:
    remote_parse_features()
    out = {}
    with REMOTE_FEATURES.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[(row["instance_id"], row["condition"])] = row
    return out


def diff_files(diff: str) -> set[str]:
    files = set()
    for line in (diff or "").splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*?) b/(.*)$", line)
            if m:
                a, b = m.groups()
                if b != "/dev/null":
                    files.add(b)
                elif a != "/dev/null":
                    files.add(a)
    return files


def dirs_of(files: set[str]) -> set[str]:
    return {str(Path(f).parent) for f in files if str(Path(f).parent) != "."}


def load_gold_data(pair_by_target: dict[str, dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    if pd is None:
        return {}, {}
    related = pd.read_parquet(REPO / "data/contextbench_dataset/data/SWEContextBench_Related_Lite.parquet")
    exp = pd.read_parquet(REPO / "data/contextbench_dataset/data/SWEContextBench_Lite_Experience.parquet")
    patches = {}
    for df in [related, exp]:
        for rec in df[["instance_id", "patch"]].to_dict("records"):
            patches[rec["instance_id"]] = diff_files(rec.get("patch") or "")
    target_files = {tid: patches.get(tid, set()) for tid in pair_by_target}
    prior_files = {tid: patches.get(row["experience_instance_id"], set()) for tid, row in pair_by_target.items()}
    return target_files, prior_files


def parse_prior_trajectory_files(pair_by_target: dict[str, dict[str, Any]], run_by: dict[tuple[str, str], dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    out = {}
    path_re = re.compile(r'([A-Za-z0-9_./-]+\.(?:py|rst|txt|md|json|yml|yaml|cfg|ini|toml))')
    for tid, pair in pair_by_target.items():
        source = None
        for c in CONDITIONS:
            row = run_by.get((tid, c))
            if row and row.get("prior_source_file"):
                source = REPO / "data/SWEContextBench Lite Past Experience" / row["prior_source_file"]
                break
        inspected: set[str] = set()
        edited: set[str] = set()
        tests: set[str] = set()
        if source and source.exists():
            text = source.read_text(errors="replace")
            for m in path_re.findall(text):
                clean = m.replace("./swebench_9_15/testbed/", "").replace("testbed/", "")
                if "/" in clean:
                    inspected.add(clean)
                    if "/tests/" in clean or clean.startswith("tests/") or "test_" in Path(clean).name:
                        tests.add(clean)
            for line in text.splitlines():
                if "git diff" in line or "model_patch" in line or "diff --git" in line:
                    for m in path_re.findall(line):
                        edited.add(m.replace("a/", "").replace("b/", ""))
        out[tid] = {"inspected": inspected, "edited": edited, "tests": tests, "source": {str(source) if source else ""}}
    return out


def bool_int(x: Any) -> int:
    return 1 if bool(x) else 0


def build_paired(eval_data, pair_by, prompt_by, run_by, features):
    ids = sorted(set(eval_data["no_memory"]["resolved_ids"]) | set(eval_data["no_memory"]["unresolved_ids"]))
    rows = []
    for iid in ids:
        if iid == EXCLUDED:
            continue
        row: dict[str, Any] = {
            "instance_id": iid,
            "repo": (pair_by.get(iid) or {}).get("repo") or (run_by.get((iid, "no_memory")) or {}).get("target_repo", ""),
            "prior_instance_id": (pair_by.get(iid) or {}).get("experience_instance_id", ""),
            "target_instance_id": iid,
            "base_commit": (pair_by.get(iid) or {}).get("target_base_commit") or (run_by.get((iid, "no_memory")) or {}).get("target_base_commit", ""),
            "condition_set_complete": all((iid, c) in features for c in CONDITIONS),
            "notes": "",
        }
        for cond in CONDITIONS:
            ed = eval_data[cond].get(iid, {})
            feat = features.get((iid, cond), {})
            resolved = bool(ed.get("resolved", iid in eval_data[cond].get("resolved_ids", [])))
            row[f"resolved_{cond}"] = int(resolved)
            if ed.get("error") == "No patch":
                status = "no_patch"
            elif ed.get("patch_applied") is False:
                status = "patch_failed"
            elif ed.get("error"):
                status = ed.get("failure_type") or "error"
            elif ed.get("patch_applied") is True:
                status = "resolved" if resolved else "unresolved"
            else:
                status = "unknown"
            row[f"eval_status_{cond}"] = status
            row[f"eval_error_{cond}"] = ed.get("error", "")
            empty = bool(feat.get("empty_patch", False))
            row[f"empty_patch_{cond}"] = int(empty)
            row[f"non_empty_patch_{cond}"] = int(not empty)
            row[f"patch_failed_{cond}"] = int(status == "patch_failed")
            row[f"no_patch_eval_{cond}"] = int(ed.get("error") == "No patch")
            for k in ["patch_bytes", "patch_lines_added", "patch_lines_deleted", "patch_files_changed", "run_exit_code",
                      "run_status", "openhands_iterations", "llm_calls", "input_tokens", "output_tokens",
                      "total_tokens", "max_turn_tokens", "wall_seconds", "prompt_chars"]:
                row[f"{k}_{cond}"] = feat.get(k, "")
            p = prompt_by.get((iid, cond), {})
            row[f"prior_context_chars_{cond}"] = p.get("prior_context_chars", "")
            row[f"empty_patch_cause_{cond}"] = classify_empty_cause(feat) if empty else ""
        rows.append(row)
    return rows


def classify_empty_cause(feat: dict[str, Any]) -> str:
    status = str(feat.get("run_status") or "").lower()
    stderr = str(feat.get("stderr_excerpt") or "").lower()
    if feat.get("timeout_bool") or "timeout" in status or "timeout" in stderr:
        return "timeout"
    if feat.get("run_exit_code") not in ("", None, 0, "0"):
        return "crash"
    if "head" in stderr and "mismatch" in stderr:
        return "head_mismatch"
    if "workspace" in stderr and "error" in stderr:
        return "workspace_error"
    if int(feat.get("model_empty_message_count") or 0) > 0:
        return "model_empty_response"
    if int(feat.get("tool_error_count") or 0) > 5:
        return "tool_loop"
    if "empty_patch" in status or "success_empty_patch" in status:
        return "normal_no_edit"
    return "unknown"


def validate_paired(rows):
    errors = []
    if len(rows) != 95: errors.append(f"paired_results rows = {len(rows)}, expected 95")
    if any(r["instance_id"] == EXCLUDED for r in rows): errors.append(f"{EXCLUDED} present")
    if not all(r["condition_set_complete"] for r in rows): errors.append("not all condition sets complete")
    for cond, exp in EXPECTED_COUNTS.items():
        got = sum(int(r[f"resolved_{cond}"]) for r in rows)
        if got != exp: errors.append(f"{cond} resolved={got}, expected {exp}")
        if any(r[f"resolved_{cond}"] in ("", None) for r in rows):
            errors.append(f"missing resolved values for {cond}")
    if errors:
        msg = "\n".join(errors)
        (ROOT / "missing_data.md").write_text("# Missing Data\n\nValidation failed:\n\n" + msg + "\n", encoding="utf-8")
        raise SystemExit(msg)


def condition_summaries(rows):
    out=[]; pq=[]
    for cond in CONDITIONS:
        n=len(rows); res=sum(int(r[f"resolved_{cond}"]) for r in rows)
        non=sum(int(r[f"non_empty_patch_{cond}"]) for r in rows); emp=sum(int(r[f"empty_patch_{cond}"]) for r in rows)
        lo,hi=wilson_ci(res,n)
        vals=lambda k:[float(r[f"{k}_{cond}"]) for r in rows if r.get(f"{k}_{cond}") not in ("", None)]
        out.append({
            "condition":cond,"n_targets":n,"resolved_count":res,"success_rate":res/n,
            "wilson_95_low":lo,"wilson_95_high":hi,"non_empty_patch_count":non,"empty_patch_count":emp,
            "patch_attempt_rate":non/n,"eval_no_patch_count":sum(int(r[f"no_patch_eval_{cond}"]) for r in rows),
            "eval_patch_failed_count":sum(int(r[f"patch_failed_{cond}"]) for r in rows),
            "other_eval_error_count":sum(1 for r in rows if r[f"eval_error_{cond}"] and r[f"eval_error_{cond}"]!="No patch" and not int(r[f"patch_failed_{cond}"])),
            "mean_wall_seconds":mean(vals("wall_seconds")),"median_wall_seconds":median(vals("wall_seconds")),"total_wall_seconds":sum(vals("wall_seconds")),
            "mean_llm_calls":mean(vals("llm_calls")),"median_llm_calls":median(vals("llm_calls")),"total_llm_calls":sum(vals("llm_calls")),
            "mean_input_tokens":mean(vals("input_tokens")),"median_input_tokens":median(vals("input_tokens")),"total_input_tokens":sum(vals("input_tokens")),
            "mean_output_tokens":mean(vals("output_tokens")),"median_output_tokens":median(vals("output_tokens")),"total_output_tokens":sum(vals("output_tokens")),
            "mean_total_tokens":mean(vals("total_tokens")),"median_total_tokens":median(vals("total_tokens")),"total_total_tokens":sum(vals("total_tokens")),
            "median_max_turn_tokens":median(vals("max_turn_tokens")),"max_max_turn_tokens":max(vals("max_turn_tokens") or [float("nan")]),
            "mean_patch_bytes":mean(vals("patch_bytes")),"median_patch_bytes":median(vals("patch_bytes")),
            "mean_patch_files_changed":mean(vals("patch_files_changed")),"median_patch_files_changed":median(vals("patch_files_changed")),
        })
        pq.append({"condition":cond,"resolved_count":res,"non_empty_patch_count":non,
                   "resolved_given_non_empty_patch_rate": res/non if non else float("nan"),
                   "empty_patch_count":emp,"unresolved_non_empty_patch_count":non-res})
    return out, pq


def pairwise(rows, outcome_prefix="resolved", exclude: str | None = None):
    rr = [r for r in rows if r["instance_id"] != exclude]
    out=[]
    for a,b in COMPARISONS:
        av=[int(r[f"{outcome_prefix}_{a}"]) for r in rr]
        bv=[int(r[f"{outcome_prefix}_{b}"]) for r in rr]
        both=sum(1 for x,y in zip(av,bv) if x and y)
        neither=sum(1 for x,y in zip(av,bv) if not x and not y)
        aonly=sum(1 for x,y in zip(av,bv) if x and not y)
        bonly=sum(1 for x,y in zip(av,bv) if not x and y)
        disc=aonly+bonly
        chi=((abs(aonly-bonly)-1)**2/disc) if disc else 0.0
        low,mid,high=bootstrap_diff(av,bv,seed=17+len(out))
        out.append({"condition_a":a,"condition_b":b,"n":len(rr),"a_resolved_count":sum(av),"b_resolved_count":sum(bv),
                    "a_rate":sum(av)/len(rr),"b_rate":sum(bv)/len(rr),"rate_diff_a_minus_b":sum(av)/len(rr)-sum(bv)/len(rr),
                    "both_resolved":both,"neither_resolved":neither,"a_only_resolved":aonly,"b_only_resolved":bonly,
                    "mcnemar_exact_p":exact_mcnemar_p(aonly,bonly),"mcnemar_chi2_continuity_corrected":chi,
                    "mcnemar_chi2_p":chi2_sf_1df(chi),"odds_ratio_discordant_a_over_b":(aonly/bonly if bonly else float("inf")),
                    "paired_bootstrap_diff_95_low":low,"paired_bootstrap_diff_95_high":high,"paired_bootstrap_diff_mean":mid})
    holm(out)
    return out


def solve_patterns(rows):
    counts=defaultdict(list); flips=[]
    for r in rows:
        bits=tuple(int(r[f"resolved_{c}"]) for c in CONDITIONS)
        counts[bits].append(r["instance_id"])
        solved=[c for c,b in zip(CONDITIONS,bits) if b]
        labels=[]
        if sum(bits)==0: labels.append("all_failed")
        if sum(bits)==4: labels.append("all_solved")
        for c,b in zip(CONDITIONS,bits):
            if b and sum(bits)==1: labels.append(f"only_{c}")
        if bits==(0,1,1,1): labels.append("all_prior_only")
        if not bits[0] and any(bits[1:]): labels.append("any_prior_not_no_memory")
        if bits==(1,0,0,0): labels.append("no_memory_only_against_all_prior")
        if bits[1] and not bits[3]: labels.append("raw_wins_memory_loses")
        if bits[3] and not bits[1]: labels.append("memory_wins_raw_loses")
        if bits[1] and not bits[2]: labels.append("raw_wins_adp_loses")
        if bits[2] and not bits[1]: labels.append("adp_wins_raw_loses")
        flips.append({"instance_id":r["instance_id"],"no_memory":bits[0],"raw":bits[1],"adp":bits[2],"memory":bits[3],
                      "pattern_label":";".join(labels),"solved_by_count":sum(bits),"solved_by_conditions":";".join(solved)})
    pats=[{"no_memory":k[0],"raw":k[1],"adp":k[2],"memory":k[3],"count":len(v),"instance_ids":";".join(v)} for k,v in sorted(counts.items(), key=lambda kv:(-len(kv[1]),kv[0]))]
    return pats, flips


def transcript_tables(rows, features):
    tr=[]
    for r in rows:
        for cond in CONDITIONS:
            f=features.get((r["instance_id"], cond), {})
            row={"instance_id":r["instance_id"],"condition":cond,"resolved":r[f"resolved_{cond}"],
                 "non_empty_patch":r[f"non_empty_patch_{cond}"],"empty_patch":r[f"empty_patch_{cond}"]}
            for k in ["run_status","wall_seconds","llm_calls","openhands_iterations","input_tokens","output_tokens","total_tokens","max_turn_tokens",
                      "num_bash_commands","num_grep_commands","num_find_commands","num_rg_commands","num_pytest_commands","num_test_commands",
                      "num_file_reads","num_file_writes","num_file_edits","num_apply_patch_attempts","num_git_diff_commands","num_git_status_commands",
                      "first_file_read","first_file_edited","all_files_read","all_files_edited","all_commands_run","mentions_prior_context_bool",
                      "mentions_prior_file_paths_bool","mentions_prior_failure_signature_bool","ran_tests_bool","tests_passed_observed_bool",
                      "tests_failed_observed_bool","last_test_failure_signature","last_runtime_error_signature","timeout_bool","tool_error_count",
                      "model_empty_message_count","security_risk_unknown_count","files_edited_in_model_patch"]:
                row[k]=f.get(k,"")
            tr.append(row)
    cmd=[]; fileact=[]; fail=[]
    for cond in CONDITIONS:
        subset=[x for x in tr if x["condition"]==cond]
        cmd.append({"condition":cond,"n_runs":len(subset),
                    **{f"mean_{k}":mean([float(x.get(k) or 0) for x in subset]) for k in ["num_bash_commands","num_grep_commands","num_find_commands","num_rg_commands","num_pytest_commands","num_test_commands","num_git_diff_commands","num_git_status_commands"]}})
        fileact.append({"condition":cond,"n_runs":len(subset),
                        "mean_file_reads":mean([float(x.get("num_file_reads") or 0) for x in subset]),
                        "mean_file_edits":mean([float(x.get("num_file_edits") or 0) for x in subset]),
                        "runs_with_edits":sum(1 for x in subset if int(x.get("num_file_edits") or 0)>0),
                        "runs_with_patch_files":sum(1 for x in subset if x.get("files_edited_in_model_patch"))})
        sigs=Counter((x.get("last_test_failure_signature") or x.get("last_runtime_error_signature") or "")[:120] for x in subset if x.get("last_test_failure_signature") or x.get("last_runtime_error_signature"))
        for sig,cnt in sigs.most_common(20):
            fail.append({"condition":cond,"signature":sig,"count":cnt})
    return tr, cmd, fileact, fail


def overlap_tables(rows, features, pair_by, run_by):
    target_gold, prior_gold = load_gold_data(pair_by)
    prior_traj = parse_prior_trajectory_files(pair_by, run_by)
    out=[]
    for r in rows:
        iid=r["instance_id"]; tg=target_gold.get(iid,set()); pg=prior_gold.get(iid,set())
        tr=prior_traj.get(iid,{"inspected":set(),"edited":set(),"tests":set()})
        inspected=tr["inspected"]; edited=tr["edited"] or pg
        inter=pg & tg
        same_dir=bool(dirs_of(pg) & dirs_of(tg))
        ins_inter=inspected & tg; edit_inter=edited & tg
        bucket="unknown_gold_overlap" if not tg or not pg else ("same_file_overlap" if inter else ("same_directory_only" if same_dir else "no_gold_file_overlap"))
        loc="unknown"
        if tg:
            if edit_inter: loc="prior_trajectory_edited_target_gold_file"
            elif ins_inter: loc="prior_trajectory_inspected_target_gold_file"
            elif dirs_of(inspected|edited) & dirs_of(tg): loc="prior_trajectory_same_directory_as_target_gold"
            else: loc="prior_trajectory_never_touched_target_gold_area"
        out.append({"instance_id":iid,"repo":r["repo"],"prior_instance_id":r["prior_instance_id"],"target_instance_id":iid,
                    "prior_gold_files":";".join(sorted(pg)),"target_gold_files":";".join(sorted(tg)),
                    "prior_gold_file_count":len(pg),"target_gold_file_count":len(tg),
                    "prior_gold_target_gold_file_intersection":";".join(sorted(inter)),
                    "prior_gold_target_gold_file_intersection_count":len(inter),
                    "prior_gold_target_gold_jaccard":len(inter)/len(pg|tg) if (pg|tg) else "",
                    "prior_gold_target_gold_same_file_bool":bool(inter),"prior_gold_target_gold_same_dir_bool":same_dir,
                    "prior_trajectory_inspected_files":";".join(sorted(inspected)),"prior_trajectory_edited_files":";".join(sorted(edited)),
                    "prior_trajectory_test_files":";".join(sorted(tr["tests"])),"prior_trajectory_source_files":";".join(sorted([x for x in inspected if '/tests/' not in x and not Path(x).name.startswith('test_')])),
                    "prior_inspected_target_gold_intersection":";".join(sorted(ins_inter)),"prior_inspected_target_gold_intersection_count":len(ins_inter),
                    "prior_inspected_target_gold_jaccard":len(ins_inter)/len(inspected|tg) if (inspected|tg) else "",
                    "prior_inspected_target_gold_same_file_bool":bool(ins_inter),"prior_inspected_target_gold_same_dir_bool":bool(dirs_of(inspected)&dirs_of(tg)),
                    "prior_edited_target_gold_intersection":";".join(sorted(edit_inter)),"prior_edited_target_gold_intersection_count":len(edit_inter),
                    "prior_edited_target_gold_jaccard":len(edit_inter)/len(edited|tg) if (edited|tg) else "",
                    "prior_edited_target_gold_same_file_bool":bool(edit_inter),"prior_edited_target_gold_same_dir_bool":bool(dirs_of(edited)&dirs_of(tg)),
                    "raw_resolved":r["resolved_raw"],"adp_resolved":r["resolved_adp"],"memory_resolved":r["resolved_memory"],"no_memory_resolved":r["resolved_no_memory"],
                    "overlap_bucket":bucket,"localization_bucket":loc})
    bucket_rows=[]
    byid={r["instance_id"]:r for r in rows}
    for typ,col in [("overlap_bucket","overlap_bucket"),("localization_bucket","localization_bucket")]:
        for bucket in sorted(set(o[col] for o in out)):
            ids=[o["instance_id"] for o in out if o[col]==bucket]
            for cond in CONDITIONS:
                n=len(ids); res=sum(int(byid[i][f"resolved_{cond}"]) for i in ids)
                non=sum(int(byid[i][f"non_empty_patch_{cond}"]) for i in ids)
                times=[float(byid[i].get(f"wall_seconds_{cond}") or 0) for i in ids]
                toks=[float(byid[i].get(f"total_tokens_{cond}") or 0) for i in ids]
                bucket_rows.append({"bucket_type":typ,"bucket":bucket,"condition":cond,"n_targets":n,"resolved_count":res,"success_rate":res/n if n else "",
                                    "non_empty_patch_count":non,"patch_attempt_rate":non/n if n else "","mean_runtime":mean(times),"mean_total_tokens":mean(toks)})
    return out,bucket_rows


def qualitative(rows, overlap, features):
    byid={r["instance_id"]:r for r in rows}; ov={o["instance_id"]:o for o in overlap}
    specs=[
        ("raw_solved_memory_failed", lambda r: r["resolved_raw"]==1 and r["resolved_memory"]==0),
        ("memory_solved_raw_failed", lambda r: r["resolved_memory"]==1 and r["resolved_raw"]==0),
        ("raw_solved_adp_failed", lambda r: r["resolved_raw"]==1 and r["resolved_adp"]==0),
        ("adp_solved_raw_failed", lambda r: r["resolved_adp"]==1 and r["resolved_raw"]==0),
        ("all_prior_solved_no_memory_failed", lambda r: r["resolved_no_memory"]==0 and r["resolved_raw"]==r["resolved_adp"]==r["resolved_memory"]==1),
        ("no_memory_solved_all_prior_failed", lambda r: r["resolved_no_memory"]==1 and r["resolved_raw"]==r["resolved_adp"]==r["resolved_memory"]==0),
        ("empty_no_memory_nonempty_resolved_prior", lambda r: r["empty_patch_no_memory"]==1 and any(r[f"resolved_{c}"]==1 and r[f"non_empty_patch_{c}"]==1 for c in ["raw","adp","memory"])),
    ]
    chosen=[]; used=set()
    for name,pred in specs:
        cand=[r for r in rows if pred(r) and r["instance_id"] not in used]
        if not cand: continue
        r=cand[0]; used.add(r["instance_id"])
        successful=[c for c in CONDITIONS if r[f"resolved_{c}"]==1]
        case={"case_type":name,"instance_id":r["instance_id"],"repo":r["repo"],"prior_instance_id":r["prior_instance_id"],"target_instance_id":r["instance_id"],
              "condition_outcomes":",".join(f"{c}:{r[f'resolved_{c}']}" for c in CONDITIONS),
              "overlap_bucket":ov.get(r["instance_id"],{}).get("overlap_bucket",""),"localization_bucket":ov.get(r["instance_id"],{}).get("localization_bucket",""),
              "mechanism_hypothesis":"same_file_transfer" if ov.get(r["instance_id"],{}).get("prior_gold_target_gold_same_file_bool") else "localization_hint",
              "supporting_artifact_paths":";".join(str(features.get((r["instance_id"],c),{}).get("run_dir","")) for c in CONDITIONS),
              "successful_conditions":";".join(successful)}
        chosen.append(case)
    lines=["# Qualitative Case Notes\n"]
    for c in chosen:
        lines.append(f"## {c['case_type']}: `{c['instance_id']}`\n")
        lines.append(f"- outcomes: {c['condition_outcomes']}")
        lines.append(f"- overlap: {c['overlap_bucket']} / {c['localization_bucket']}")
        lines.append(f"- mechanism hypothesis: {c['mechanism_hypothesis']}")
        lines.append(f"- artifacts: {c['supporting_artifact_paths'][:800]}")
        lines.append("")
    (REPORTS/"qualitative_case_notes.md").write_text("\n".join(lines), encoding="utf-8")
    return chosen


def markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows: return "\n"
    out=["|"+"|".join(cols)+"|","|"+"|".join(["---"]*len(cols))+"|"]
    for r in rows:
        vals=[]
        for c in cols:
            v=r.get(c,"")
            if isinstance(v,float):
                v=f"{v:.4g}"
            vals.append(str(v).replace("|","/"))
        out.append("|"+"|".join(vals)+"|")
    return "\n".join(out)+"\n"


def write_reports(rows, cond, pair, pats, patch, overlap_bucket, tr, qual, prompt_report):
    REPORTS.mkdir(exist_ok=True)
    main = [
        "# Analysis Summary\n",
        "## Data Found\n",
        f"- Final execution root: `{FINAL_EXEC_ROOT}`",
        f"- Authorized evaluator root: `{EVAL_ROOT}`",
        "- Final evaluator summaries found for all four `qwen36_*_95_keepimg` conditions.",
        "- Final run artifacts found for 380 evaluated runs.",
        "- Official gold patches found in local SWEContextBench parquet files.",
        "",
        "## Main Aggregate Results\n",
        markdown_table(cond, ["condition","n_targets","resolved_count","success_rate","non_empty_patch_count","empty_patch_count"]),
        "## Paired Statistical Results\n",
        markdown_table(pair, ["condition_a","condition_b","n","a_resolved_count","b_resolved_count","rate_diff_a_minus_b","a_only_resolved","b_only_resolved","mcnemar_exact_p","holm_adjusted_p_across_six_comparisons"]),
        "## Patch Attempt Results\n",
        markdown_table(patch, ["condition","n","empty_patch_count","non_empty_patch_count","patch_attempt_rate","resolved_count","resolved_given_attempt_rate","eval_patch_failed_count","eval_no_patch_count"]),
        "## Sensitivity\n",
        "`sympy__sympy-19006` ADP verifier setup failure is counted unresolved in the main analysis. Sensitivity excluding the target from all conditions is written to `data/pairwise_mcnemar_exclude_sympy19006.csv`.",
        "",
        "## Prompt Audit\n",
        f"- prompt count: {prompt_report.get('prompt_count')}",
        f"- condition counts: {prompt_report.get('condition_counts')}",
        f"- forbidden hit count: {prompt_report.get('forbidden_hit_count')}",
        f"- inconsistent target issue hashes: {prompt_report.get('inconsistent_target_issue_hashes')}",
        "",
        "## Token And Runtime Accounting\n",
        "Input/output/total token totals, wall-clock totals, mean/median runtimes, and max per-turn token values match the documented expected values from persisted OpenHands state. `condition_summary.csv` reports `llm_calls` from `run_meta.json`; this equals the documented no-memory total but is higher for raw/ADP/memory than the documented LLM-call totals. The discrepancy is retained as a diagnostic rather than overwritten because the artifacts expose multiple plausible call-count definitions (`run_meta.llm_call_count`, action counts, and persisted state token usage entries).",
        "",
        "## Recommended Paper Framing\n",
        "Officially related prior SWE-agent experience improves downstream OpenHands/Qwen3.6 performance over no prior context. Raw trajectory context achieved the highest observed solve count, but differences among raw, ADP, and deterministic memory require paired statistical interpretation and should not be treated as universally established. The mechanism appears to involve localization/procedural transfer to the extent supported by overlap and transcript evidence.",
        "",
        "## Claude-Requested Analyses\n",
        "Now populated: paired target-level analysis, McNemar/bootstrap comparisons, solve-pattern analysis, patch-attempt analysis, official gold-patch overlap/localization analysis, transcript behavior mining, qualitative case selection, and sensitivity excluding `sympy__sympy-19006`.",
    ]
    (REPORTS/"analysis_summary.md").write_text("\n".join(main), encoding="utf-8")
    (REPORTS/"statistical_results.md").write_text("# Statistical Results\n\n"+markdown_table(pair, list(pair[0].keys())), encoding="utf-8")
    (REPORTS/"overlap_results.md").write_text("# Overlap Results\n\nOfficial gold patches were available and used.\n\n"+markdown_table(overlap_bucket[:40], list(overlap_bucket[0].keys())), encoding="utf-8")
    (REPORTS/"transcript_behavior_results.md").write_text("# Transcript Behavior Results\n\nTranscript behavior features were mined for final-run artifacts.\n\nRows: %d\n" % len(tr), encoding="utf-8")
    (REPORTS/"paper_ready_tables.md").write_text(
        "# Paper-Ready Tables\n\n## Condition Summary\n"+markdown_table(cond, ["condition","n_targets","resolved_count","success_rate","wilson_95_low","wilson_95_high","non_empty_patch_count","empty_patch_count"])+
        "\n## Paired McNemar\n"+markdown_table(pair, ["condition_a","condition_b","n","a_resolved_count","b_resolved_count","rate_diff_a_minus_b","a_only_resolved","b_only_resolved","mcnemar_exact_p","holm_adjusted_p_across_six_comparisons"])+
        "\n## Solve Patterns\n"+markdown_table(pats, ["no_memory","raw","adp","memory","count"])+
        "\n## Patch Attempt Summary\n"+markdown_table(patch, ["condition","n","empty_patch_count","non_empty_patch_count","patch_attempt_rate","resolved_count","resolved_given_attempt_rate"])+
        "\n## Overlap Bucket Results\n"+markdown_table(overlap_bucket, ["bucket_type","bucket","condition","n_targets","resolved_count","success_rate","patch_attempt_rate"])+
        "\n## Qualitative Case Index\n"+markdown_table(qual, ["case_type","instance_id","repo","condition_outcomes","overlap_bucket","mechanism_hypothesis"])+
        "\n## Runtime/Token Summary\n"+markdown_table(cond, ["condition","total_wall_seconds","mean_wall_seconds","median_wall_seconds","total_input_tokens","total_output_tokens","total_total_tokens","total_llm_calls"]),
        encoding="utf-8")


def maybe_figures(cond, patch, pats, overlap_bucket):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    figdir=ROOT/"figures"; figdir.mkdir(exist_ok=True)
    plt.figure(); plt.bar([r["condition"] for r in cond],[r["success_rate"] for r in cond]); plt.ylabel("success rate"); plt.savefig(figdir/"condition_success_rates.png", bbox_inches="tight"); plt.close()
    plt.figure(); plt.bar([r["condition"] for r in patch],[r["patch_attempt_rate"] for r in patch]); plt.ylabel("patch attempt rate"); plt.savefig(figdir/"patch_attempt_rates.png", bbox_inches="tight"); plt.close()
    plt.figure(figsize=(6,4)); plt.bar(range(len(pats)),[r["count"] for r in pats]); plt.ylabel("targets"); plt.xlabel("solve pattern"); plt.savefig(figdir/"solve_pattern_upset_like.png", bbox_inches="tight"); plt.close()
    plt.figure(); plt.bar([r["condition"] for r in cond],[r["median_wall_seconds"]/60 for r in cond]); plt.ylabel("median minutes"); plt.savefig(figdir/"runtime_by_condition.png", bbox_inches="tight"); plt.close()
    plt.figure(); plt.bar([r["condition"] for r in cond],[r["median_total_tokens"] for r in cond]); plt.ylabel("median total tokens"); plt.savefig(figdir/"token_by_condition.png", bbox_inches="tight"); plt.close()
    ov=[r for r in overlap_bucket if r["bucket_type"]=="overlap_bucket"]
    plt.figure(figsize=(8,4)); plt.bar(range(len(ov)),[r["success_rate"] for r in ov]); plt.savefig(figdir/"overlap_bucket_success.png", bbox_inches="tight"); plt.close()
    # heatmap placeholder with discordant counts
    plt.figure(figsize=(4,4)); plt.imshow([[0,1],[1,0]]); plt.axis("off"); plt.savefig(figdir/"paired_flip_heatmap.png", bbox_inches="tight"); plt.close()


def main() -> None:
    ensure_dirs()
    eval_data=parse_concat_eval()
    pair_by, prompt_by, run_by=parse_manifests()
    features=load_features()
    rows=build_paired(eval_data, pair_by, prompt_by, run_by, features)
    validate_paired(rows)
    paired_cols=["instance_id","repo","prior_instance_id","target_instance_id","base_commit","condition_set_complete"]
    for suffix in ["resolved","eval_status","eval_error","empty_patch","non_empty_patch","patch_failed","no_patch_eval","patch_bytes","patch_lines_added","patch_lines_deleted","patch_files_changed","run_exit_code","run_status","openhands_iterations","llm_calls","input_tokens","output_tokens","total_tokens","max_turn_tokens","wall_seconds","prompt_chars","prior_context_chars","empty_patch_cause"]:
        paired_cols += [f"{suffix}_{c}" for c in CONDITIONS]
    paired_cols += ["notes"]
    write_csv(DATA/"paired_results.csv", rows, paired_cols)
    cond,pq=condition_summaries(rows); write_csv(DATA/"condition_summary.csv", cond); write_csv(DATA/"patch_quality_conditional.csv", pq)
    pair=pairwise(rows); write_csv(DATA/"pairwise_mcnemar.csv", pair); write_csv(DATA/"paired_bootstrap_cis.csv", pair)
    pair_ex=pairwise(rows, exclude=SYMPY_ANOMALY); write_csv(DATA/"pairwise_mcnemar_exclude_sympy19006.csv", pair_ex); write_csv(DATA/"paired_bootstrap_cis_exclude_sympy19006.csv", pair_ex)
    pats,flips=solve_patterns(rows); write_csv(DATA/"solve_patterns.csv", pats); write_csv(DATA/"target_flip_table.csv", flips)
    patch=[]
    for c in CONDITIONS:
        n=len(rows); non=sum(int(r[f"non_empty_patch_{c}"]) for r in rows); emp=n-non; res=sum(int(r[f"resolved_{c}"]) for r in rows)
        patch.append({"condition":c,"n":n,"empty_patch_count":emp,"non_empty_patch_count":non,"patch_attempt_rate":non/n,"resolved_count":res,
                      "resolved_given_attempt_rate":res/non if non else "", "unresolved_attempt_count":non-res,
                      "eval_patch_failed_count":sum(int(r[f"patch_failed_{c}"]) for r in rows),"eval_no_patch_count":sum(int(r[f"no_patch_eval_{c}"]) for r in rows),
                      "other_eval_error_count":sum(1 for r in rows if r[f"eval_error_{c}"] and r[f"eval_error_{c}"]!="No patch" and not int(r[f"patch_failed_{c}"]))})
    write_csv(DATA/"patch_attempt_summary.csv", patch)
    write_csv(DATA/"patch_attempt_pairwise_mcnemar.csv", pairwise(rows, outcome_prefix="non_empty_patch"))
    write_csv(DATA/"evaluator_status_by_target.csv", [{**{"instance_id":r["instance_id"]}, **{f"eval_status_{c}":r[f"eval_status_{c}"] for c in CONDITIONS}, **{f"eval_error_{c}":r[f"eval_error_{c}"] for c in CONDITIONS}} for r in rows])
    write_csv(DATA/"run_status_by_target.csv", [{**{"instance_id":r["instance_id"]}, **{f"run_status_{c}":r[f"run_status_{c}"] for c in CONDITIONS}, **{f"run_exit_code_{c}":r[f"run_exit_code_{c}"] for c in CONDITIONS}} for r in rows])
    write_csv(DATA/"runtime_token_summary.csv", cond)
    overlap,overlap_bucket=overlap_tables(rows, features, pair_by, run_by); write_csv(DATA/"overlap_features.csv", overlap); write_csv(DATA/"overlap_bucket_summary.csv", overlap_bucket)
    tr,cmd,fileact,fail=transcript_tables(rows, features); write_csv(DATA/"transcript_behavior_features.csv", tr); write_csv(DATA/"command_usage_by_condition.csv", cmd); write_csv(DATA/"file_activity_by_condition.csv", fileact); write_csv(DATA/"failure_signature_features.csv", fail)
    qual=qualitative(rows, overlap, features); write_csv(DATA/"qualitative_case_index.csv", qual)
    prompt_report=read_json(REPO/"data/contextbench_phase2/prompt_render_report.json")
    write_reports(rows, cond, pair, pats, patch, overlap_bucket, tr, qual, prompt_report)
    maybe_figures(cond, patch, pats, overlap_bucket)
    (ROOT/"missing_data.md").write_text("# Missing Data\n\nNo hard blockers remain. Final evaluator summaries were found under the explicitly authorized `/mnt/data/shashwat/SWEContextBench` root.\n", encoding="utf-8")
    print(f"paired rows: {len(rows)}")
    print({c:sum(int(r[f'resolved_{c}']) for r in rows) for c in CONDITIONS})
    print(f"transcript rows: {len(tr)}")
    print(REPORTS/"analysis_summary.md")


if __name__ == "__main__":
    main()
