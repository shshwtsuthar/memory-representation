#!/usr/bin/env python3
"""Paired statistics for the ContextBench posthoc analysis."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


CONDITIONS = ["no_memory", "raw", "adp", "memory"]
PAIRS = [
    ("raw", "no_memory"),
    ("memory", "no_memory"),
    ("adp", "no_memory"),
    ("raw", "memory"),
    ("raw", "adp"),
    ("memory", "adp"),
]
SYMPY_ANOMALY = "sympy__sympy-19006"


PAIRWISE_COLUMNS = [
    "condition_a",
    "condition_b",
    "n",
    "a_resolved_count",
    "b_resolved_count",
    "a_rate",
    "b_rate",
    "rate_diff_a_minus_b",
    "both_resolved",
    "neither_resolved",
    "a_only_resolved",
    "b_only_resolved",
    "mcnemar_exact_p",
    "mcnemar_chi2_continuity_corrected",
    "mcnemar_chi2_p",
    "odds_ratio_discordant_a_over_b",
    "paired_bootstrap_diff_95_low",
    "paired_bootstrap_diff_95_high",
    "paired_bootstrap_diff_mean",
    "holm_adjusted_p_across_six_comparisons",
]


BOOTSTRAP_COLUMNS = [
    "condition_a",
    "condition_b",
    "n",
    "paired_bootstrap_diff_mean",
    "paired_bootstrap_diff_95_low",
    "paired_bootstrap_diff_95_high",
    "seed",
    "n_boot",
]


SOLVE_PATTERN_COLUMNS = ["no_memory", "raw", "adp", "memory", "count", "instance_ids"]
TARGET_FLIP_COLUMNS = [
    "instance_id",
    "no_memory",
    "raw",
    "adp",
    "memory",
    "pattern_label",
    "solved_by_count",
    "solved_by_conditions",
]


PATCH_ATTEMPT_COLUMNS = [
    "condition",
    "n",
    "empty_patch_count",
    "non_empty_patch_count",
    "patch_attempt_rate",
    "resolved_count",
    "resolved_given_attempt_rate",
    "unresolved_attempt_count",
    "eval_patch_failed_count",
    "eval_no_patch_count",
    "other_eval_error_count",
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


RUNTIME_TOKEN_COLUMNS = [
    "condition",
    "n_targets",
    "observed_total_input_tokens",
    "observed_total_output_tokens",
    "observed_total_tokens",
    "observed_total_llm_calls",
    "observed_total_wall_seconds",
    "observed_total_wall_hours",
    "observed_mean_runtime_minutes",
    "observed_median_runtime_minutes",
    "observed_median_max_turn_tokens",
    "observed_max_turn_tokens",
    "expected_total_input_tokens",
    "expected_total_output_tokens",
    "expected_total_tokens",
    "expected_total_llm_calls",
    "expected_total_wall_hours",
    "expected_mean_runtime_minutes",
    "expected_median_runtime_minutes",
    "expected_median_max_turn_tokens",
    "expected_max_turn_tokens",
    "verification_status",
    "notes",
]


EXPECTED_RUNTIME = {
    "no_memory": {
        "input": 149_116_755,
        "output": 4_677_579,
        "total": 153_794_334,
        "llm_calls": 4_416,
        "wall_h": 45.20,
        "mean_m": 28.5,
        "median_m": 13.3,
        "median_max": 39_370,
        "max_max": 98_304,
    },
    "raw": {
        "input": 154_667_541,
        "output": 3_310_983,
        "total": 157_978_524,
        "llm_calls": 3_079,
        "wall_h": 34.86,
        "mean_m": 22.0,
        "median_m": 12.4,
        "median_max": 48_160,
        "max_max": 98_307,
    },
    "adp": {
        "input": 177_160_048,
        "output": 4_886_695,
        "total": 182_046_743,
        "llm_calls": 2_880,
        "wall_h": 45.29,
        "mean_m": 28.6,
        "median_m": 14.1,
        "median_max": 51_035,
        "max_max": 98_306,
    },
    "memory": {
        "input": 147_791_422,
        "output": 3_843_866,
        "total": 151_635_288,
        "llm_calls": 2_780,
        "wall_h": 41.17,
        "mean_m": 26.0,
        "median_m": 11.3,
        "median_max": 45_419,
        "max_max": 98_305,
    },
}


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
            writer.writerow({col: row.get(col, "") for col in columns})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def is_complete_binary(rows: list[dict[str, str]], prefix: str) -> bool:
    for row in rows:
        for cond in CONDITIONS:
            if row.get(f"{prefix}_{cond}", "") not in {"0", "1"}:
                return False
    return bool(rows)


def int_values(rows: list[dict[str, str]], key: str) -> list[int]:
    vals: list[int] = []
    for row in rows:
        v = row.get(key, "")
        if v in {"0", "1"}:
            vals.append(int(v))
    return vals


def numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        v = row.get(key, "")
        if v in ("", None):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    return vals


def mean(vals: list[float]) -> str:
    return "" if not vals else f"{statistics.fmean(vals):.6g}"


def median(vals: list[float]) -> str:
    return "" if not vals else f"{statistics.median(vals):.6g}"


def total(vals: list[float]) -> str:
    return "" if not vals else f"{sum(vals):.6g}"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
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


def chi2_p_value(stat: float) -> str:
    try:
        from scipy.stats import chi2  # type: ignore

        return f"{float(chi2.sf(stat, 1)):.6g}"
    except Exception:
        return ""


def paired_bootstrap_diff(a: list[int], b: list[int], n_boot: int = 20000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(a)
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    diffs.sort()
    lo = diffs[int(0.025 * (n_boot - 1))]
    mid = statistics.fmean(diffs)
    hi = diffs[int(0.975 * (n_boot - 1))]
    return lo, mid, hi


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    indexed = [(i, float(row["mcnemar_exact_p"])) for i, row in enumerate(rows)]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    adjusted = [1.0] * m
    running = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        adjusted[idx] = running
    for idx, value in enumerate(adjusted):
        rows[idx]["holm_adjusted_p_across_six_comparisons"] = f"{value:.6g}"


def pairwise_table(
    rows: list[dict[str, str]],
    *,
    prefix: str,
    n_boot: int = 20000,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    if not is_complete_binary(rows, prefix):
        return out, boot_rows
    for a, b in PAIRS:
        av = [int(r[f"{prefix}_{a}"]) for r in rows]
        bv = [int(r[f"{prefix}_{b}"]) for r in rows]
        n = len(rows)
        both = sum(1 for x, y in zip(av, bv) if x == 1 and y == 1)
        neither = sum(1 for x, y in zip(av, bv) if x == 0 and y == 0)
        a_only = sum(1 for x, y in zip(av, bv) if x == 1 and y == 0)
        b_only = sum(1 for x, y in zip(av, bv) if x == 0 and y == 1)
        exact = exact_mcnemar_p(a_only, b_only)
        discord = a_only + b_only
        chi2_stat = ((abs(a_only - b_only) - 1) ** 2 / discord) if discord else 0.0
        odds = "" if b_only == 0 else f"{a_only / b_only:.6g}"
        if b_only == 0 and a_only > 0:
            odds = "inf"
        lo, boot_mean, hi = paired_bootstrap_diff(av, bv, n_boot=n_boot, seed=seed)
        row = {
            "condition_a": a,
            "condition_b": b,
            "n": n,
            "a_resolved_count": sum(av),
            "b_resolved_count": sum(bv),
            "a_rate": f"{sum(av) / n:.6g}",
            "b_rate": f"{sum(bv) / n:.6g}",
            "rate_diff_a_minus_b": f"{sum(av) / n - sum(bv) / n:.6g}",
            "both_resolved": both,
            "neither_resolved": neither,
            "a_only_resolved": a_only,
            "b_only_resolved": b_only,
            "mcnemar_exact_p": f"{exact:.6g}",
            "mcnemar_chi2_continuity_corrected": f"{chi2_stat:.6g}",
            "mcnemar_chi2_p": chi2_p_value(chi2_stat),
            "odds_ratio_discordant_a_over_b": odds,
            "paired_bootstrap_diff_95_low": f"{lo:.6g}",
            "paired_bootstrap_diff_95_high": f"{hi:.6g}",
            "paired_bootstrap_diff_mean": f"{boot_mean:.6g}",
            "holm_adjusted_p_across_six_comparisons": "",
        }
        out.append(row)
        boot_rows.append(
            {
                "condition_a": a,
                "condition_b": b,
                "n": n,
                "paired_bootstrap_diff_mean": f"{boot_mean:.6g}",
                "paired_bootstrap_diff_95_low": f"{lo:.6g}",
                "paired_bootstrap_diff_95_high": f"{hi:.6g}",
                "seed": seed,
                "n_boot": n_boot,
            }
        )
    holm_adjust(out)
    return out, boot_rows


def label_pattern(values: dict[str, int]) -> str:
    solved = [c for c in CONDITIONS if values[c] == 1]
    s = set(solved)
    if len(s) == 0:
        return "all_failed"
    if len(s) == 4:
        return "all_solved"
    if s == {"no_memory"}:
        return "only_no_memory"
    if s == {"raw"}:
        return "only_raw"
    if s == {"adp"}:
        return "only_adp"
    if s == {"memory"}:
        return "only_memory"
    if s == {"raw", "adp", "memory"}:
        return "all_prior_only"
    labels = []
    if values["no_memory"] == 0 and any(values[c] for c in ["raw", "adp", "memory"]):
        labels.append("any_prior_not_no_memory")
    if values["no_memory"] == 1 and not any(values[c] for c in ["raw", "adp", "memory"]):
        labels.append("no_memory_only_against_all_prior")
    if values["raw"] == 1 and values["memory"] == 0:
        labels.append("raw_wins_memory_loses")
    if values["memory"] == 1 and values["raw"] == 0:
        labels.append("memory_wins_raw_loses")
    if values["raw"] == 1 and values["adp"] == 0:
        labels.append("raw_wins_adp_loses")
    if values["adp"] == 1 and values["raw"] == 0:
        labels.append("adp_wins_raw_loses")
    return ";".join(labels) if labels else "+".join(solved)


def solve_patterns(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not is_complete_binary(rows, "resolved"):
        return [], []
    buckets: dict[tuple[int, int, int, int], list[str]] = {}
    target_rows: list[dict[str, Any]] = []
    for row in rows:
        vals = {cond: int(row[f"resolved_{cond}"]) for cond in CONDITIONS}
        key = tuple(vals[c] for c in CONDITIONS)
        buckets.setdefault(key, []).append(row["instance_id"])
        solved = [c for c in CONDITIONS if vals[c] == 1]
        target_rows.append(
            {
                "instance_id": row["instance_id"],
                **{c: vals[c] for c in CONDITIONS},
                "pattern_label": label_pattern(vals),
                "solved_by_count": len(solved),
                "solved_by_conditions": ";".join(solved),
            }
        )
    pattern_rows = []
    for key, instance_ids in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        pattern_rows.append(
            {
                "no_memory": key[0],
                "raw": key[1],
                "adp": key[2],
                "memory": key[3],
                "count": len(instance_ids),
                "instance_ids": ";".join(sorted(instance_ids)),
            }
        )
    return pattern_rows, target_rows


def condition_summary(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    patch_quality: list[dict[str, Any]] = []
    patch_attempt_summary: list[dict[str, Any]] = []
    resolved_complete = is_complete_binary(rows, "resolved")
    for cond in CONDITIONS:
        n = len(rows)
        res = int_values(rows, f"resolved_{cond}")
        attempts = int_values(rows, f"non_empty_patch_{cond}")
        empties = int_values(rows, f"empty_patch_{cond}")
        has_attempts = len(attempts) == n
        resolved_count = sum(res) if resolved_complete else ""
        if resolved_complete:
            low, high = wilson_ci(sum(res), n)
        else:
            low, high = (float("nan"), float("nan"))
        metrics = {m: numeric_values(rows, f"{m}_{cond}") for m in [
            "wall_seconds",
            "llm_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "max_turn_tokens",
            "patch_bytes",
            "patch_files_changed",
        ]}
        eval_patch_failed = sum(int(r.get(f"patch_failed_{cond}", "0")) for r in rows if r.get(f"patch_failed_{cond}", "") in {"0", "1"})
        eval_no_patch = sum(int(r.get(f"no_patch_eval_{cond}", "0")) for r in rows if r.get(f"no_patch_eval_{cond}", "") in {"0", "1"})
        other_eval = sum(1 for r in rows if r.get(f"eval_error_{cond}", "") and not r.get(f"patch_failed_{cond}", "") == "1" and not r.get(f"no_patch_eval_{cond}", "") == "1")
        summary.append(
            {
                "condition": cond,
                "n_targets": n,
                "resolved_count": resolved_count,
                "success_rate": f"{sum(res) / n:.6g}" if resolved_complete and n else "",
                "wilson_95_low": f"{low:.6g}" if resolved_complete else "",
                "wilson_95_high": f"{high:.6g}" if resolved_complete else "",
                "non_empty_patch_count": sum(attempts) if has_attempts else "",
                "empty_patch_count": sum(empties) if len(empties) == n else "",
                "patch_attempt_rate": f"{sum(attempts) / n:.6g}" if has_attempts and n else "",
                "eval_no_patch_count": eval_no_patch if eval_no_patch else "",
                "eval_patch_failed_count": eval_patch_failed if eval_patch_failed else "",
                "other_eval_error_count": other_eval if other_eval else "",
                "mean_wall_seconds": mean(metrics["wall_seconds"]),
                "median_wall_seconds": median(metrics["wall_seconds"]),
                "total_wall_seconds": total(metrics["wall_seconds"]),
                "mean_llm_calls": mean(metrics["llm_calls"]),
                "median_llm_calls": median(metrics["llm_calls"]),
                "total_llm_calls": total(metrics["llm_calls"]),
                "mean_input_tokens": mean(metrics["input_tokens"]),
                "median_input_tokens": median(metrics["input_tokens"]),
                "total_input_tokens": total(metrics["input_tokens"]),
                "mean_output_tokens": mean(metrics["output_tokens"]),
                "median_output_tokens": median(metrics["output_tokens"]),
                "total_output_tokens": total(metrics["output_tokens"]),
                "mean_total_tokens": mean(metrics["total_tokens"]),
                "median_total_tokens": median(metrics["total_tokens"]),
                "total_total_tokens": total(metrics["total_tokens"]),
                "median_max_turn_tokens": median(metrics["max_turn_tokens"]),
                "max_max_turn_tokens": f"{max(metrics['max_turn_tokens']):.6g}" if metrics["max_turn_tokens"] else "",
                "mean_patch_bytes": mean(metrics["patch_bytes"]),
                "median_patch_bytes": median(metrics["patch_bytes"]),
                "mean_patch_files_changed": mean(metrics["patch_files_changed"]),
                "median_patch_files_changed": median(metrics["patch_files_changed"]),
            }
        )
        resolved_given_attempt = ""
        unresolved_attempts = ""
        if resolved_complete and has_attempts and sum(attempts):
            resolved_attempts = sum(1 for r in rows if r[f"resolved_{cond}"] == "1" and r[f"non_empty_patch_{cond}"] == "1")
            unresolved_attempts = sum(1 for r in rows if r[f"resolved_{cond}"] == "0" and r[f"non_empty_patch_{cond}"] == "1")
            resolved_given_attempt = f"{resolved_attempts / sum(attempts):.6g}"
        patch_quality.append(
            {
                "condition": cond,
                "resolved_count": resolved_count,
                "non_empty_patch_count": sum(attempts) if has_attempts else "",
                "resolved_given_non_empty_patch_rate": resolved_given_attempt,
                "empty_patch_count": sum(empties) if len(empties) == n else "",
                "unresolved_non_empty_patch_count": unresolved_attempts,
            }
        )
        patch_attempt_summary.append(
            {
                "condition": cond,
                "n": n,
                "empty_patch_count": sum(empties) if len(empties) == n else "",
                "non_empty_patch_count": sum(attempts) if has_attempts else "",
                "patch_attempt_rate": f"{sum(attempts) / n:.6g}" if has_attempts and n else "",
                "resolved_count": resolved_count,
                "resolved_given_attempt_rate": resolved_given_attempt,
                "unresolved_attempt_count": unresolved_attempts,
                "eval_patch_failed_count": eval_patch_failed if eval_patch_failed else "",
                "eval_no_patch_count": eval_no_patch if eval_no_patch else "",
                "other_eval_error_count": other_eval if other_eval else "",
            }
        )
    return summary, patch_quality, patch_attempt_summary


def runtime_token_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cond in CONDITIONS:
        wall = numeric_values(rows, f"wall_seconds_{cond}")
        max_turn = numeric_values(rows, f"max_turn_tokens_{cond}")
        input_tokens = numeric_values(rows, f"input_tokens_{cond}")
        output_tokens = numeric_values(rows, f"output_tokens_{cond}")
        total_tokens = numeric_values(rows, f"total_tokens_{cond}")
        llm_calls = numeric_values(rows, f"llm_calls_{cond}")
        exp = EXPECTED_RUNTIME[cond]
        has_observed = bool(wall or input_tokens or output_tokens or total_tokens or llm_calls or max_turn)
        notes = []
        if not has_observed:
            notes.append("Full run token/runtime artifacts were unavailable locally.")
        notes.append(
            "Known max per-turn token targets around 98K exceed the nominal 65K context label; cause cannot be determined without OpenHands/provider accounting artifacts."
        )
        out.append(
            {
                "condition": cond,
                "n_targets": len(rows),
                "observed_total_input_tokens": f"{sum(input_tokens):.0f}" if input_tokens else "",
                "observed_total_output_tokens": f"{sum(output_tokens):.0f}" if output_tokens else "",
                "observed_total_tokens": f"{sum(total_tokens):.0f}" if total_tokens else "",
                "observed_total_llm_calls": f"{sum(llm_calls):.0f}" if llm_calls else "",
                "observed_total_wall_seconds": f"{sum(wall):.6g}" if wall else "",
                "observed_total_wall_hours": f"{sum(wall) / 3600:.6g}" if wall else "",
                "observed_mean_runtime_minutes": f"{statistics.fmean(wall) / 60:.6g}" if wall else "",
                "observed_median_runtime_minutes": f"{statistics.median(wall) / 60:.6g}" if wall else "",
                "observed_median_max_turn_tokens": f"{statistics.median(max_turn):.6g}" if max_turn else "",
                "observed_max_turn_tokens": f"{max(max_turn):.6g}" if max_turn else "",
                "expected_total_input_tokens": exp["input"],
                "expected_total_output_tokens": exp["output"],
                "expected_total_tokens": exp["total"],
                "expected_total_llm_calls": exp["llm_calls"],
                "expected_total_wall_hours": exp["wall_h"],
                "expected_mean_runtime_minutes": exp["mean_m"],
                "expected_median_runtime_minutes": exp["median_m"],
                "expected_median_max_turn_tokens": exp["median_max"],
                "expected_max_turn_tokens": exp["max_max"],
                "verification_status": "verified" if has_observed else "unverified_missing_artifacts",
                "notes": " ".join(notes),
            }
        )
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")).replace("\n", " ") for c in columns) + " |")
    return "\n".join(out)


def write_placeholder_figures(out_dir: Path, rows: list[dict[str, str]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "condition_success_rates.png",
        "patch_attempt_rates.png",
        "paired_flip_heatmap.png",
        "solve_pattern_upset_like.png",
        "runtime_by_condition.png",
        "token_by_condition.png",
    ]
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        for name in names:
            plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, "Data unavailable" if not is_complete_binary(rows, "resolved") else name, ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(fig_dir / name, dpi=140)
            plt.close()
        if is_complete_binary(rows, "resolved"):
            rates = [sum(int(r[f"resolved_{c}"]) for r in rows) / len(rows) for c in CONDITIONS]
            plt.figure(figsize=(6, 3.5))
            plt.bar(CONDITIONS, rates)
            plt.ylabel("success rate")
            plt.ylim(0, max(0.25, max(rates) * 1.2))
            plt.tight_layout()
            plt.savefig(fig_dir / "condition_success_rates.png", dpi=140)
            plt.close()
            patterns, _ = solve_patterns(rows)
            plt.figure(figsize=(8, max(3, 0.35 * len(patterns))))
            labels = [f"{r['no_memory']}{r['raw']}{r['adp']}{r['memory']}" for r in patterns]
            counts = [int(r["count"]) for r in patterns]
            plt.barh(labels, counts)
            plt.xlabel("target count")
            plt.ylabel("pattern no/raw/adp/mem")
            plt.tight_layout()
            plt.savefig(fig_dir / "solve_pattern_upset_like.png", dpi=140)
            plt.close()
        if is_complete_binary(rows, "non_empty_patch"):
            rates = [sum(int(r[f"non_empty_patch_{c}"]) for r in rows) / len(rows) for c in CONDITIONS]
            plt.figure(figsize=(6, 3.5))
            plt.bar(CONDITIONS, rates)
            plt.ylabel("non-empty patch rate")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(fig_dir / "patch_attempt_rates.png", dpi=140)
            plt.close()
    except Exception as exc:  # noqa: BLE001
        for name in names:
            (fig_dir / (name + ".txt")).write_text(f"Figure unavailable: {exc}\n", encoding="utf-8")


def write_reports(
    out_dir: Path,
    rows: list[dict[str, str]],
    pairwise: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
    patch_attempt_rows: list[dict[str, Any]],
    patch_attempt_pairwise: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> None:
    complete = is_complete_binary(rows, "resolved")
    lines = ["# Statistical Results", ""]
    if not complete:
        missing = sum(1 for r in rows for c in CONDITIONS if r.get(f"resolved_{c}", "") not in {"0", "1"})
        lines.append(f"Paired statistical tests were not run because `paired_results.csv` lacks complete evaluator truth ({missing} missing condition-target outcomes).")
    else:
        lines.extend(
            [
                "## Pairwise McNemar Comparisons",
                markdown_table(pairwise, PAIRWISE_COLUMNS),
                "",
                "Exact two-sided McNemar p-values are primary. Holm adjustment is across the six planned comparisons.",
                "",
                "## Sensitivity Excluding sympy__sympy-19006",
                markdown_table(sensitivity, PAIRWISE_COLUMNS),
                "",
                "## Solve Patterns",
                markdown_table(patterns, SOLVE_PATTERN_COLUMNS),
            ]
        )
    lines.extend(
        [
            "",
            "## Patch Attempt Analysis",
            markdown_table(patch_attempt_rows, PATCH_ATTEMPT_COLUMNS),
            "",
            "## Patch Attempt Pairwise McNemar",
            markdown_table(patch_attempt_pairwise, PAIRWISE_COLUMNS),
            "",
            "## Runtime / Token Accounting",
            markdown_table(
                runtime_rows,
                [
                    "condition",
                    "verification_status",
                    "observed_total_tokens",
                    "expected_total_tokens",
                    "observed_total_llm_calls",
                    "expected_total_llm_calls",
                    "observed_total_wall_hours",
                    "expected_total_wall_hours",
                    "expected_max_turn_tokens",
                ],
            ),
            "",
            "The expected max per-turn token values are around 98K despite the 65K context configuration label. This run did not have the full OpenHands accounting artifacts needed to determine whether that reflects tokenizer mismatch, prompt+completion accounting, rolling context, or actual served context.",
        ]
    )
    write_text(out_dir / "reports/statistical_results.md", "\n".join(lines) + "\n")

    paper = [
        "# Paper-Ready Tables",
        "",
        "## Paired McNemar Comparisons",
        markdown_table(pairwise, PAIRWISE_COLUMNS) if pairwise else "_Unavailable: missing evaluator truth._",
        "",
        "## Solve-Pattern Counts",
        markdown_table(patterns, SOLVE_PATTERN_COLUMNS) if patterns else "_Unavailable: missing evaluator truth._",
        "",
        "## Patch-Attempt Summary",
        markdown_table(patch_attempt_rows, PATCH_ATTEMPT_COLUMNS),
        "",
        "## Runtime/Token Summary",
        markdown_table(
            runtime_rows,
            [
                "condition",
                "verification_status",
                "expected_total_input_tokens",
                "expected_total_output_tokens",
                "expected_total_tokens",
                "expected_total_llm_calls",
                "expected_total_wall_hours",
                "expected_median_max_turn_tokens",
                "expected_max_turn_tokens",
            ],
        ),
    ]
    write_text(out_dir / "reports/paper_ready_tables.md", "\n".join(paper) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    rows = read_csv(out_dir / "data/paired_results.csv")
    eprint(f"[stats] loaded {len(rows)} paired rows")

    summary, patch_quality, patch_attempt_summary = condition_summary(rows)
    runtime_rows = runtime_token_summary(rows)
    write_csv(out_dir / "data/condition_summary.csv", summary, CONDITION_SUMMARY_COLUMNS)
    write_csv(out_dir / "data/patch_quality_conditional.csv", patch_quality, PATCH_QUALITY_COLUMNS)
    write_csv(out_dir / "data/patch_attempt_summary.csv", patch_attempt_summary, PATCH_ATTEMPT_COLUMNS)
    write_csv(out_dir / "data/runtime_token_summary.csv", runtime_rows, RUNTIME_TOKEN_COLUMNS)

    pairwise, boot = pairwise_table(rows, prefix="resolved", n_boot=args.n_boot, seed=args.seed)
    write_csv(out_dir / "data/pairwise_mcnemar.csv", pairwise, PAIRWISE_COLUMNS)
    write_csv(out_dir / "data/paired_bootstrap_cis.csv", boot, BOOTSTRAP_COLUMNS)

    sensitivity_rows = [r for r in rows if r.get("instance_id") != SYMPY_ANOMALY]
    sens_pairwise, sens_boot = pairwise_table(sensitivity_rows, prefix="resolved", n_boot=args.n_boot, seed=args.seed)
    write_csv(out_dir / "data/pairwise_mcnemar_exclude_sympy19006.csv", sens_pairwise, PAIRWISE_COLUMNS)
    write_csv(out_dir / "data/paired_bootstrap_cis_exclude_sympy19006.csv", sens_boot, BOOTSTRAP_COLUMNS)

    patterns, flips = solve_patterns(rows)
    write_csv(out_dir / "data/solve_patterns.csv", patterns, SOLVE_PATTERN_COLUMNS)
    write_csv(out_dir / "data/target_flip_table.csv", flips, TARGET_FLIP_COLUMNS)

    patch_pairwise, _patch_boot = pairwise_table(rows, prefix="non_empty_patch", n_boot=args.n_boot, seed=args.seed)
    # Rename resolved-count columns semantically would break requested schema, so keep generic pairwise McNemar columns.
    write_csv(out_dir / "data/patch_attempt_pairwise_mcnemar.csv", patch_pairwise, PAIRWISE_COLUMNS)

    write_placeholder_figures(out_dir, rows)
    write_reports(out_dir, rows, pairwise, sens_pairwise, patterns, patch_attempt_summary, patch_pairwise, runtime_rows)
    if not is_complete_binary(rows, "resolved"):
        eprint("[stats] evaluator truth incomplete; wrote schema-correct empty paired-stat outputs")
        return 2
    eprint("[stats] completed paired statistics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
