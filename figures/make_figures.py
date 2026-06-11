#!/usr/bin/env python3
"""Generate publication figures for the memory-representation study.

The script is intentionally defensive: it validates the paper-level counts
before plotting so regenerated figures cannot silently drift away from the
analysis CSVs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any

# Avoid noisy cache warnings on shared machines where ~/.config is not writable.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openhands_adp_memory_mplconfig")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REQUIRED_FILES = {
    "condition_summary": "condition_summary.csv",
    "paired_bootstrap_cis": "paired_bootstrap_cis.csv",
    "pairwise_mcnemar": "pairwise_mcnemar.csv",
    "patch_attempt_pairwise_mcnemar": "patch_attempt_pairwise_mcnemar.csv",
    "patch_attempt_summary": "patch_attempt_summary.csv",
    "solve_patterns": "solve_patterns.csv",
    "paired_results": "paired_results.csv",
    "target_flip_table": "target_flip_table.csv",
    "overlap_bucket_summary": "overlap_bucket_summary.csv",
    "overlap_features": "overlap_features.csv",
    "time_to_gold_file": "time_to_gold_file.csv",
    "tool_timeline_events": "tool_timeline_events.csv",
    "representation_disagreement_cases": "representation_disagreement_cases.csv",
    "evidence_target_overlap": "evidence_target_overlap.csv",
    "representation_evidence_inventory": "representation_evidence_inventory.csv",
    "tool_pattern_by_outcome": "tool_pattern_by_outcome.csv",
    "transcript_behavior_features": "transcript_behavior_features.csv",
}

OPTIONAL_FILES = {
    "qualitative_case_index": "qualitative_case_index.csv",
}

CONDITIONS = ["no_memory", "raw", "adp", "memory"]
PRIOR_CONDITIONS = ["raw", "adp", "memory"]
LABELS = {
    "no_memory": "None",
    "raw": "Trace",
    "adp": "Action",
    "memory": "Digest",
}
SHORT_LABELS = {
    "no_memory": "None",
    "raw": "Trace",
    "adp": "Action",
    "memory": "Digest",
}

# Okabe-Ito inspired, with a neutral for no prior context.
COLORS = {
    "no_memory": "#4D4D4D",
    "raw": "#0072B2",
    "adp": "#D55E00",
    "memory": "#009E73",
    "patch": "#CC79A7",
    "gray": "#BDBDBD",
    "dark_gray": "#666666",
}
MARKERS = {
    "no_memory": "o",
    "raw": "s",
    "adp": "^",
    "memory": "D",
}

N_TARGETS = 95


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "lines.linewidth": 1.4,
            "axes.axisbelow": True,
        }
    )


def fail(message: str) -> None:
    raise AssertionError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in REQUIRED_FILES.values() if not (data_dir / name).exists()]
    if missing:
        fail(
            "Missing required input CSV(s) in "
            f"{data_dir}: {', '.join(sorted(missing))}"
        )

    dfs: dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_FILES.items():
        dfs[key] = pd.read_csv(data_dir / filename, low_memory=False)

    for key, filename in OPTIONAL_FILES.items():
        path = data_dir / filename
        if path.exists():
            dfs[key] = pd.read_csv(path, low_memory=False)
    return dfs


def require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{table_name} is missing required columns: {missing}")


def pattern_count(solve_patterns: pd.DataFrame, values: tuple[int, int, int, int]) -> int:
    mask = (
        (solve_patterns["no_memory"] == values[0])
        & (solve_patterns["raw"] == values[1])
        & (solve_patterns["adp"] == values[2])
        & (solve_patterns["memory"] == values[3])
    )
    return int(solve_patterns.loc[mask, "count"].sum())


def validate_counts(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    condition_summary = dfs["condition_summary"]
    require_columns(
        condition_summary,
        ["condition", "n_targets", "resolved_count", "non_empty_patch_count"],
        "condition_summary.csv",
    )
    expected_condition_counts = {
        "no_memory": {"resolved_count": 10, "non_empty_patch_count": 62},
        "raw": {"resolved_count": 19, "non_empty_patch_count": 82},
        "adp": {"resolved_count": 15, "non_empty_patch_count": 81},
        "memory": {"resolved_count": 16, "non_empty_patch_count": 79},
    }
    require(set(condition_summary["condition"]) == set(CONDITIONS), "Unexpected conditions in condition_summary.csv")
    condition_counts: dict[str, dict[str, int]] = {}
    for condition, expected in expected_condition_counts.items():
        row = condition_summary.loc[condition_summary["condition"] == condition]
        require(len(row) == 1, f"Expected exactly one condition_summary row for {condition}")
        row = row.iloc[0]
        require(int(row["n_targets"]) == N_TARGETS, f"{condition} n_targets is not {N_TARGETS}")
        condition_counts[condition] = {
            "n_targets": int(row["n_targets"]),
            "resolved_count": int(row["resolved_count"]),
            "non_empty_patch_count": int(row["non_empty_patch_count"]),
        }
        for column, expected_value in expected.items():
            observed = int(row[column])
            require(
                observed == expected_value,
                f"condition_summary {condition} {column}: expected {expected_value}, observed {observed}",
            )

    patch_summary = dfs["patch_attempt_summary"]
    require_columns(
        patch_summary,
        ["condition", "n", "empty_patch_count", "non_empty_patch_count", "resolved_count"],
        "patch_attempt_summary.csv",
    )
    for condition, expected in expected_condition_counts.items():
        row = patch_summary.loc[patch_summary["condition"] == condition]
        require(len(row) == 1, f"Expected exactly one patch_attempt_summary row for {condition}")
        row = row.iloc[0]
        require(int(row["n"]) == N_TARGETS, f"{condition} patch_attempt_summary n is not {N_TARGETS}")
        require(
            int(row["non_empty_patch_count"]) == expected["non_empty_patch_count"],
            f"patch_attempt_summary {condition} non-empty count mismatch",
        )
        require(
            int(row["resolved_count"]) == expected["resolved_count"],
            f"patch_attempt_summary {condition} resolved count mismatch",
        )

    solve_patterns = dfs["solve_patterns"]
    require_columns(
        solve_patterns,
        ["no_memory", "raw", "adp", "memory", "count", "instance_ids"],
        "solve_patterns.csv",
    )
    require(int(solve_patterns["count"].sum()) == N_TARGETS, "solve_patterns counts do not sum to 95")
    solve_pattern_counts = {
        "all_failed": pattern_count(solve_patterns, (0, 0, 0, 0)),
        "all_four_solved": pattern_count(solve_patterns, (1, 1, 1, 1)),
        "raw_only": pattern_count(solve_patterns, (0, 1, 0, 0)),
        "memory_only": pattern_count(solve_patterns, (0, 0, 0, 1)),
        "adp_only": pattern_count(solve_patterns, (0, 0, 1, 0)),
        "no_memory_only_all_prior_fail": pattern_count(solve_patterns, (1, 0, 0, 0)),
    }
    expected_pattern_counts = {
        "all_failed": 66,
        "all_four_solved": 5,
        "raw_only": 6,
        "memory_only": 3,
        "adp_only": 2,
        "no_memory_only_all_prior_fail": 1,
    }
    for key, expected in expected_pattern_counts.items():
        require(
            solve_pattern_counts[key] == expected,
            f"solve_patterns {key}: expected {expected}, observed {solve_pattern_counts[key]}",
        )

    paired_results = dfs["paired_results"]
    require_columns(
        paired_results,
        [
            "instance_id",
            "condition_set_complete",
            "resolved_no_memory",
            "resolved_raw",
            "resolved_adp",
            "resolved_memory",
            "eval_status_adp",
        ],
        "paired_results.csv",
    )
    require(len(paired_results) == N_TARGETS, "paired_results.csv does not contain 95 rows")
    require(
        paired_results["instance_id"].nunique() == N_TARGETS,
        "paired_results.csv does not contain 95 unique targets",
    )
    require(
        "django__django-28147" not in set(paired_results["instance_id"]),
        "Excluded target django__django-28147 is present in paired_results.csv",
    )
    sympy = paired_results.loc[paired_results["instance_id"] == "sympy__sympy-19006"]
    require(len(sympy) == 1, "sympy__sympy-19006 is missing from paired_results.csv")
    sympy = sympy.iloc[0]
    require(
        int(sympy["resolved_adp"]) == 0 and "verifier" in str(sympy["eval_status_adp"]).lower(),
        "sympy__sympy-19006 Action/ADP verifier anomaly is not counted unresolved",
    )

    oracle_prior = int(
        paired_results[["resolved_raw", "resolved_adp", "resolved_memory"]].max(axis=1).sum()
    )
    oracle_all = int(
        paired_results[
            ["resolved_no_memory", "resolved_raw", "resolved_adp", "resolved_memory"]
        ]
        .max(axis=1)
        .sum()
    )
    best_fixed_prior = max(
        condition_counts["raw"]["resolved_count"],
        condition_counts["adp"]["resolved_count"],
        condition_counts["memory"]["resolved_count"],
    )
    oracle_headroom = oracle_prior - best_fixed_prior
    require(oracle_prior == 28, f"oracle_prior expected 28, observed {oracle_prior}")
    require(oracle_all == 29, f"oracle_all expected 29, observed {oracle_all}")
    require(oracle_headroom == 9, f"oracle headroom expected 9, observed {oracle_headroom}")

    for key, table_name in [
        ("paired_bootstrap_cis", "paired_bootstrap_cis.csv"),
        ("pairwise_mcnemar", "pairwise_mcnemar.csv"),
        ("patch_attempt_pairwise_mcnemar", "patch_attempt_pairwise_mcnemar.csv"),
    ]:
        require_columns(
            dfs[key],
            [
                "condition_a",
                "condition_b",
                "n",
                "rate_diff_a_minus_b",
                "mcnemar_exact_p",
                "paired_bootstrap_diff_95_low",
                "paired_bootstrap_diff_95_high",
                "holm_adjusted_p_across_six_comparisons",
            ],
            table_name,
        )
        require(len(dfs[key]) == 6, f"{table_name} should contain six pairwise rows")

    time_to_gold = dfs["time_to_gold_file"]
    require_columns(
        time_to_gold,
        [
            "instance_id",
            "condition",
            "first_gold_file_read_step",
            "first_gold_file_edit_step",
            "num_search_commands",
            "edited_target_gold_file_bool",
            "ran_test_bool",
            "patch_touches_gold_file_bool",
            "resolved",
            "empty_patch",
        ],
        "time_to_gold_file.csv",
    )
    require(len(time_to_gold) == 380, "time_to_gold_file.csv does not contain 380 rows")
    require(
        time_to_gold["instance_id"].nunique() == N_TARGETS,
        "time_to_gold_file.csv does not contain 95 unique targets",
    )
    per_condition = time_to_gold.groupby("condition")["instance_id"].count().to_dict()
    require(
        per_condition == {c: N_TARGETS for c in CONDITIONS},
        f"time_to_gold_file.csv does not have 95 rows per condition: {per_condition}",
    )

    events = dfs["tool_timeline_events"]
    require_columns(
        events,
        [
            "instance_id",
            "condition",
            "step_index",
            "event_type",
            "tool_name",
            "normalized_command_type",
            "tool_error_bool",
        ],
        "tool_timeline_events.csv",
    )
    require(len(events) > 0, "tool_timeline_events.csv is empty")

    overlap_summary = dfs["overlap_bucket_summary"]
    require_columns(
        overlap_summary,
        ["bucket_type", "bucket", "condition", "n_targets", "resolved_count", "success_rate"],
        "overlap_bucket_summary.csv",
    )
    overlap_bucket_counts = (
        overlap_summary.loc[overlap_summary["bucket_type"] == "overlap_bucket"]
        .drop_duplicates("bucket")
        .set_index("bucket")["n_targets"]
        .astype(int)
        .to_dict()
    )
    localization_bucket_counts = (
        overlap_summary.loc[overlap_summary["bucket_type"] == "localization_bucket"]
        .drop_duplicates("bucket")
        .set_index("bucket")["n_targets"]
        .astype(int)
        .to_dict()
    )
    require(
        sum(overlap_bucket_counts.values()) == N_TARGETS,
        f"Overlap bucket sizes do not sum to 95: {overlap_bucket_counts}",
    )
    require(
        sum(localization_bucket_counts.values()) == N_TARGETS,
        f"Localization bucket sizes do not sum to 95: {localization_bucket_counts}",
    )

    evidence = dfs["evidence_target_overlap"]
    inventory = dfs["representation_evidence_inventory"]
    require(
        len(evidence) == 285 and len(inventory) == 285,
        "Evidence CSVs should contain 95 targets x 3 prior representations",
    )

    return {
        "condition_counts": condition_counts,
        "patch_attempt_counts": {
            c: {
                "non_empty_patch_count": int(
                    patch_summary.loc[patch_summary["condition"] == c, "non_empty_patch_count"].iloc[0]
                ),
                "empty_patch_count": int(
                    patch_summary.loc[patch_summary["condition"] == c, "empty_patch_count"].iloc[0]
                ),
            }
            for c in CONDITIONS
        },
        "solve_pattern_counts": solve_pattern_counts,
        "oracle_prior": oracle_prior,
        "oracle_all": oracle_all,
        "best_fixed_prior": best_fixed_prior,
        "oracle_headroom_over_best_fixed_prior": oracle_headroom,
        "time_to_gold_rows": int(len(time_to_gold)),
        "timeline_event_rows": int(len(events)),
        "overlap_bucket_counts": overlap_bucket_counts,
        "localization_bucket_counts": localization_bucket_counts,
        "excluded_target": "django__django-28147",
        "adp_verifier_anomaly_counted_unresolved": "sympy__sympy-19006",
    }


def pair_row(df: pd.DataFrame, condition_a: str, condition_b: str) -> pd.Series:
    direct = df.loc[
        (df["condition_a"] == condition_a) & (df["condition_b"] == condition_b)
    ]
    if len(direct) == 1:
        return direct.iloc[0]
    reverse = df.loc[
        (df["condition_a"] == condition_b) & (df["condition_b"] == condition_a)
    ]
    if len(reverse) == 1:
        row = reverse.iloc[0].copy()
        for col in [
            "rate_diff_a_minus_b",
            "paired_bootstrap_diff_95_low",
            "paired_bootstrap_diff_95_high",
            "paired_bootstrap_diff_mean",
        ]:
            if col in row:
                row[col] = -float(row[col])
        low = float(row["paired_bootstrap_diff_95_low"])
        high = float(row["paired_bootstrap_diff_95_high"])
        row["paired_bootstrap_diff_95_low"] = min(low, high)
        row["paired_bootstrap_diff_95_high"] = max(low, high)
        return row
    fail(f"Missing pairwise row for {condition_a} vs {condition_b}")
    raise RuntimeError("unreachable")


def p_text(p_value: float) -> str:
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def p_label(name: str, p_value: float) -> str:
    text = p_text(p_value)
    if text.startswith("<"):
        return f"{name}{text}"
    return f"{name}={text}"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def wrap_label(text: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, str]:
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return {"filename": pdf.name, "png_filename": png.name}


def plot_paired_effects(dfs: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Any]:
    resolution = dfs["paired_bootstrap_cis"]
    patch = dfs["patch_attempt_pairwise_mcnemar"]
    rows = PRIOR_CONDITIONS
    y = np.arange(len(rows))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, constrained_layout=True)
    plot_specs = [
        (axes[0], resolution, "Final resolution", COLORS["raw"]),
        (axes[1], patch, "Non-empty patch attempt", COLORS["patch"]),
    ]
    figure_counts: dict[str, list[dict[str, Any]]] = {"resolved": [], "non_empty_patch": []}

    for ax, table, title, color in plot_specs:
        xs, lows, highs, labels = [], [], [], []
        for condition in rows:
            row = pair_row(table, condition, "no_memory")
            xs.append(float(row["rate_diff_a_minus_b"]) * 100)
            lows.append(float(row["paired_bootstrap_diff_95_low"]) * 100)
            highs.append(float(row["paired_bootstrap_diff_95_high"]) * 100)
            labels.append(
                f"{p_label('p', float(row['mcnemar_exact_p']))}; "
                f"{p_label('Holm', float(row['holm_adjusted_p_across_six_comparisons']))}"
            )
            target = "resolved" if table is resolution else "non_empty_patch"
            figure_counts[target].append(
                {
                    "comparison": f"{LABELS[condition]} - None",
                    "rate_diff_percentage_points": float(row["rate_diff_a_minus_b"]) * 100,
                    "ci_low_percentage_points": float(row["paired_bootstrap_diff_95_low"]) * 100,
                    "ci_high_percentage_points": float(row["paired_bootstrap_diff_95_high"]) * 100,
                    "mcnemar_exact_p": float(row["mcnemar_exact_p"]),
                    "holm_adjusted_p": float(row["holm_adjusted_p_across_six_comparisons"]),
                }
            )

        xs = np.array(xs)
        lows = np.array(lows)
        highs = np.array(highs)
        xerr = np.vstack([xs - lows, highs - xs])
        ax.errorbar(
            xs,
            y,
            xerr=xerr,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=3,
            markersize=5,
            zorder=3,
        )
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("Paired difference vs None (p.p.)")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{LABELS[c]} - None" for c in rows])
        xmin = min(-3.0, float(lows.min()) - 2.0)
        xmax = float(highs.max()) + 11.0
        ax.set_xlim(xmin, xmax)
        for yi, high, label in zip(y, highs, labels):
            ax.text(high + 1.0, yi, label, va="center", fontsize=6.4, color="#333333")

    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    files = save_figure(fig, out_dir, "fig_paired_effects")
    return {
        "figure": "Figure 1",
        **files,
        "source_csvs": [
            "paired_bootstrap_cis.csv",
            "patch_attempt_pairwise_mcnemar.csv",
        ],
        "exact_counts_used": figure_counts,
        "caption_takeaway": (
            "Resolution gains versus None are descriptive and paired-success tests are not "
            "Holm-significant; non-empty patch-attempt effects are large and Holm-significant."
        ),
        "caveats": [
            "Paired differences are not causal mediation estimates.",
            "Holm p-values are the values supplied in the pairwise CSVs.",
        ],
    }


def intersection_label(row: pd.Series) -> str:
    active = [LABELS[c] for c in CONDITIONS if int(row[c]) == 1]
    return " + ".join(active) if active else "All failed"


def plot_solve_complementarity_oracle(
    dfs: dict[str, pd.DataFrame], out_dir: Path
) -> dict[str, Any]:
    solve_patterns = dfs["solve_patterns"].copy()
    paired = dfs["paired_results"]
    solved = solve_patterns.loc[
        solve_patterns[CONDITIONS].sum(axis=1) > 0
    ].copy()
    solved["label"] = solved.apply(intersection_label, axis=1)
    solved = solved.sort_values(
        by=["count", "raw", "memory", "adp", "no_memory"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    fig = plt.figure(figsize=(7.3, 3.8), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    left = outer[0].subgridspec(2, 1, height_ratios=[2.0, 1.15], hspace=0.02)
    ax_bar = fig.add_subplot(left[0])
    ax_matrix = fig.add_subplot(left[1], sharex=ax_bar)
    ax_ladder = fig.add_subplot(outer[1])

    x = np.arange(len(solved))
    counts = solved["count"].astype(int).to_numpy()
    ax_bar.bar(x, counts, color="#8E8E8E", width=0.72)
    for xi, count in zip(x, counts):
        ax_bar.text(xi, count + 0.18, str(int(count)), ha="center", va="bottom", fontsize=7)
    ax_bar.set_ylabel("Solved targets")
    ax_bar.set_title("Solved-set intersections")
    ax_bar.set_ylim(0, max(counts) + 2.0)
    ax_bar.tick_params(axis="x", labelbottom=False)
    all_failed = pattern_count(solve_patterns, (0, 0, 0, 0))
    ax_bar.text(
        0.98,
        0.84,
        f"All failed: {all_failed}",
        transform=ax_bar.transAxes,
        ha="right",
        va="center",
        fontsize=8,
        color=COLORS["dark_gray"],
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F2F2F2", "edgecolor": "#DDDDDD"},
    )
    panel_label(ax_bar, "A")

    matrix_y = np.arange(len(CONDITIONS))[::-1]
    for idx, condition in enumerate(CONDITIONS):
        yi = matrix_y[idx]
        ax_matrix.scatter(x, np.full_like(x, yi), s=15, color="#D6D6D6", zorder=1)
    for xi, (_, row) in zip(x, solved.iterrows()):
        active_y = []
        for idx, condition in enumerate(CONDITIONS):
            yi = matrix_y[idx]
            if int(row[condition]) == 1:
                active_y.append(yi)
                ax_matrix.scatter(
                    [xi],
                    [yi],
                    s=24,
                    color=COLORS[condition],
                    edgecolor="#222222",
                    linewidth=0.35,
                    zorder=3,
                )
        if len(active_y) > 1:
            ax_matrix.plot([xi, xi], [min(active_y), max(active_y)], color="#555555", linewidth=0.8, zorder=2)
    ax_matrix.set_yticks(matrix_y)
    ax_matrix.set_yticklabels([LABELS[c] for c in CONDITIONS])
    ax_matrix.set_xlabel("Intersection pattern, sorted by count")
    ax_matrix.set_ylim(-0.6, len(CONDITIONS) - 0.4)
    ax_matrix.set_xticks([])
    ax_matrix.grid(False)

    fixed_counts = {
        "None": int(paired["resolved_no_memory"].sum()),
        "Trace": int(paired["resolved_raw"].sum()),
        "Action": int(paired["resolved_adp"].sum()),
        "Digest": int(paired["resolved_memory"].sum()),
    }
    oracle_prior = int(paired[["resolved_raw", "resolved_adp", "resolved_memory"]].max(axis=1).sum())
    oracle_all = int(
        paired[["resolved_no_memory", "resolved_raw", "resolved_adp", "resolved_memory"]]
        .max(axis=1)
        .sum()
    )
    ladder_labels = [
        "None",
        "Trace\n(best fixed prior)",
        "Prior oracle\n(retrospective)",
        "All oracle\n(retrospective)",
    ]
    ladder_values = [fixed_counts["None"], fixed_counts["Trace"], oracle_prior, oracle_all]
    lx = np.arange(len(ladder_values))
    ax_ladder.plot(lx, ladder_values, color="#333333", linewidth=1.2, zorder=1)
    ax_ladder.scatter(
        lx,
        ladder_values,
        s=48,
        color=[COLORS["no_memory"], COLORS["raw"], "#7A5195", "#333333"],
        edgecolor="#222222",
        linewidth=0.4,
        zorder=3,
    )
    for xi, value in zip(lx, ladder_values):
        ax_ladder.text(xi, value + 0.8, f"{value}/95", ha="center", va="bottom", fontsize=8)
    ax_ladder.annotate(
        "",
        xy=(2, oracle_prior),
        xytext=(1, fixed_counts["Trace"]),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "#7A5195"},
    )
    ax_ladder.text(1.5, (oracle_prior + fixed_counts["Trace"]) / 2 + 1.0, "+9", color="#7A5195", ha="center")
    ax_ladder.set_xticks(lx)
    ax_ladder.set_xticklabels(ladder_labels)
    ax_ladder.set_ylabel("Solved targets")
    ax_ladder.set_ylim(0, 33)
    ax_ladder.set_title("Retrospective oracle headroom")
    panel_label(ax_ladder, "B")

    files = save_figure(fig, out_dir, "fig_solve_complementarity_oracle")
    intersections = [
        {
            "pattern": row["label"],
            "bit_pattern": {c: int(row[c]) for c in CONDITIONS},
            "count": int(row["count"]),
        }
        for _, row in solved.iterrows()
    ]
    return {
        "figure": "Figure 2",
        **files,
        "source_csvs": ["solve_patterns.csv", "paired_results.csv", "target_flip_table.csv"],
        "exact_counts_used": {
            "intersections": intersections,
            "all_failed": all_failed,
            "ladder": {
                "none": fixed_counts["None"],
                "best_fixed_prior_trace": fixed_counts["Trace"],
                "retrospective_prior_context_oracle": oracle_prior,
                "retrospective_all_condition_oracle": oracle_all,
                "oracle_headroom_over_trace": oracle_prior - fixed_counts["Trace"],
            },
        },
        "caption_takeaway": (
            "Trace has the highest fixed prior-context solve count, but different representations "
            "solve different targets; the +9 prior-context oracle headroom is retrospective."
        ),
        "caveats": [
            "The retrospective oracle is not a deployable method.",
            "The oracle is not labeled as a method or leaderboard entry.",
        ],
    }


def plot_bucket_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    bucket_order: list[str],
    bucket_labels: dict[str, str],
    title: str,
) -> dict[str, Any]:
    bucket_counts = (
        summary.drop_duplicates("bucket").set_index("bucket")["n_targets"].astype(int).to_dict()
    )
    counts_for_manifest: dict[str, dict[str, Any]] = {}
    x = np.arange(len(bucket_order))
    for condition in CONDITIONS:
        rows = summary.loc[summary["condition"] == condition].set_index("bucket")
        rates = [float(rows.loc[b, "success_rate"]) * 100 for b in bucket_order]
        resolved = [int(rows.loc[b, "resolved_count"]) for b in bucket_order]
        ns = [int(rows.loc[b, "n_targets"]) for b in bucket_order]
        ax.plot(
            x,
            rates,
            marker=MARKERS[condition],
            color=COLORS[condition],
            linewidth=1.0,
            markersize=4.5,
            label=LABELS[condition],
        )
        counts_for_manifest[condition] = {
            b: {"resolved_count": r, "n_targets": n, "success_rate": rate / 100}
            for b, r, n, rate in zip(bucket_order, resolved, ns, rates)
        }
    tick_labels = [
        f"{bucket_labels[b]}\n(n={bucket_counts[b]})"
        for b in bucket_order
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, 32)
    ax.set_ylabel("Resolved (%)")
    ax.set_title(title)
    return counts_for_manifest


def ecdf_all_denominator(values: pd.Series, denominator: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.sort(values.dropna().astype(float).to_numpy())
    if len(finite) == 0:
        return np.array([0.0]), np.array([0.0])
    x = np.concatenate([[0.0], finite])
    y = np.concatenate([[0.0], np.arange(1, len(finite) + 1) / denominator])
    return x, y


def plot_localization_map(dfs: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Any]:
    summary = dfs["overlap_bucket_summary"]
    time_to_gold = dfs["time_to_gold_file"]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.25), constrained_layout=True)
    overlap_order = ["no_gold_file_overlap", "same_directory_only", "same_file_overlap"]
    overlap_labels = {
        "no_gold_file_overlap": "No gold-file\noverlap",
        "same_directory_only": "Same directory\nonly",
        "same_file_overlap": "Same file\noverlap",
    }
    overlap_counts = plot_bucket_panel(
        axes[0],
        summary.loc[summary["bucket_type"] == "overlap_bucket"],
        overlap_order,
        overlap_labels,
        "Gold-patch overlap",
    )
    panel_label(axes[0], "A")

    localization_order = [
        "prior_trajectory_edited_target_gold_file",
        "prior_trajectory_inspected_target_gold_file",
        "prior_trajectory_same_directory_as_target_gold",
        "prior_trajectory_never_touched_target_gold_area",
    ]
    localization_labels = {
        "prior_trajectory_edited_target_gold_file": "Prior edited\nfile",
        "prior_trajectory_inspected_target_gold_file": "Prior read\nfile",
        "prior_trajectory_same_directory_as_target_gold": "Same gold\ndir",
        "prior_trajectory_never_touched_target_gold_area": "Never\ntouched",
    }
    localization_counts = plot_bucket_panel(
        axes[1],
        summary.loc[summary["bucket_type"] == "localization_bucket"],
        localization_order,
        localization_labels,
        "Prior trajectory localization",
    )
    axes[1].set_ylabel("")
    panel_label(axes[1], "B")

    ax = axes[2]
    max_step = float(time_to_gold["first_gold_file_read_step"].max())
    ecdf_summary: dict[str, Any] = {}
    never_lines: list[str] = []
    for condition in CONDITIONS:
        rows = time_to_gold.loc[time_to_gold["condition"] == condition]
        x, y = ecdf_all_denominator(rows["first_gold_file_read_step"], N_TARGETS)
        if len(x) > 1 and x[-1] < max_step:
            x = np.append(x, max_step)
            y = np.append(y, y[-1])
        ax.step(
            x,
            y * 100,
            where="post",
            color=COLORS[condition],
            linewidth=1.5,
            label=LABELS[condition],
        )
        never = int(rows["first_gold_file_read_step"].isna().sum())
        median = float(rows["first_gold_file_read_step"].median())
        ecdf_summary[condition] = {
            "read_count": int(rows["first_gold_file_read_step"].notna().sum()),
            "never_read_count": never,
            "median_first_gold_file_read_step": median,
            "mean_search_commands": float(rows["num_search_commands"].mean()),
        }
        never_lines.append(f"{LABELS[condition]} {never}/95")
    ax.set_xlabel("Tool step")
    ax.set_ylabel("Runs that have read a target gold file (%)")
    ax.set_title("First target-gold-file read")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max_step)
    search_lines = [
        f"{LABELS[c]} {ecdf_summary[c]['mean_search_commands']:.1f}"
        for c in CONDITIONS
    ]
    ax.text(
        0.04,
        0.05,
        "Mean searches\n" + "\n".join(search_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#DDDDDD"},
    )
    ax.text(
        0.98,
        0.16,
        "Never read\n" + "\n".join(never_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#DDDDDD"},
    )
    panel_label(ax, "C")

    handles = [
        Line2D([0], [0], color=COLORS[c], marker=MARKERS[c], linewidth=1.2, markersize=4, label=LABELS[c])
        for c in CONDITIONS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))

    files = save_figure(fig, out_dir, "fig_localization_map")
    return {
        "figure": "Figure 3",
        **files,
        "source_csvs": [
            "overlap_bucket_summary.csv",
            "overlap_features.csv",
            "time_to_gold_file.csv",
        ],
        "exact_counts_used": {
            "overlap_buckets": overlap_counts,
            "localization_buckets": localization_counts,
            "ecdf_denominator_per_condition": N_TARGETS,
            "ecdf_summary": ecdf_summary,
        },
        "caption_takeaway": (
            "Prior-context gains concentrate when prior and target are local, and prior-context "
            "runs reach target-gold files earlier with fewer mean search commands, consistent "
            "with localization/procedural transfer."
        ),
        "caveats": [
            "The ECDF denominator is all 95 runs per condition; missing read steps are treated as never reached.",
            "Mechanism labels are post hoc and do not prove causality.",
        ],
    }


def plot_patch_action_funnel(dfs: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Any]:
    time_to_gold = dfs["time_to_gold_file"]
    condition_summary = dfs["condition_summary"].set_index("condition")
    patch_summary = dfs["patch_attempt_summary"].set_index("condition")
    stages = [
        ("All\nruns", lambda c, rows: N_TARGETS),
        ("Non-empty\npatch", lambda c, rows: int(condition_summary.loc[c, "non_empty_patch_count"])),
        ("Patch touches\ngold file", lambda c, rows: int(rows["patch_touches_gold_file_bool"].sum())),
        ("Edited\ngold file", lambda c, rows: int(rows["edited_target_gold_file_bool"].sum())),
        ("Ran\ntests", lambda c, rows: int(rows["ran_test_bool"].sum())),
        ("Resolved", lambda c, rows: int(condition_summary.loc[c, "resolved_count"])),
    ]
    stage_counts: dict[str, dict[str, int]] = {}

    fig, ax = plt.subplots(figsize=(7.2, 3.15), constrained_layout=True)
    x = np.arange(len(stages))
    label_offsets = {"no_memory": -8, "raw": 7, "adp": -14, "memory": 13}
    for condition in CONDITIONS:
        rows = time_to_gold.loc[time_to_gold["condition"] == condition]
        counts = [fn(condition, rows) for _, fn in stages]
        require(
            counts[1] == int(patch_summary.loc[condition, "non_empty_patch_count"]),
            f"Non-empty patch count mismatch in action funnel for {condition}",
        )
        require(
            counts[-1] == int(rows["resolved"].sum()),
            f"Resolved count mismatch in action funnel for {condition}",
        )
        stage_counts[condition] = {
            stage_name.replace("\n", " "): int(count)
            for (stage_name, _), count in zip(stages, counts)
        }
        ax.plot(
            x,
            counts,
            marker=MARKERS[condition],
            color=COLORS[condition],
            linewidth=1.3,
            markersize=5,
            label=LABELS[condition],
        )
        for xi, count in zip(x, counts):
            if xi == 0 and condition != "no_memory":
                continue
            ax.annotate(
                str(count),
                (xi, count),
                textcoords="offset points",
                xytext=(0, 8 if xi == 0 else label_offsets[condition]),
                ha="center",
                va="center",
                fontsize=6.2,
                color=COLORS[condition],
            )
    ax.set_xticks(x)
    ax.set_xticklabels([stage for stage, _ in stages])
    ax.set_ylabel("Runs out of 95")
    ax.set_ylim(0, 103)
    ax.set_title("Action funnel across repair trajectories")
    ax.legend(loc="upper right", ncol=2, frameon=False)

    files = save_figure(fig, out_dir, "fig_patch_action_funnel")
    return {
        "figure": "Figure 4",
        **files,
        "source_csvs": [
            "time_to_gold_file.csv",
            "condition_summary.csv",
            "patch_attempt_summary.csv",
        ],
        "exact_counts_used": stage_counts,
        "caption_takeaway": (
            "Prior-context conditions mainly reduce empty-patch behavior and increase target-relevant "
            "actions; final resolution is a narrower endpoint."
        ),
        "caveats": [
            "Ran tests is not strictly sequential after editing, so this is an action funnel rather than a causal pipeline.",
        ],
    }


def short_instance(instance_id: str) -> str:
    if "__" in instance_id:
        return instance_id.split("__", 1)[1]
    return instance_id


def parse_outcomes(condition_outcomes: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for piece in str(condition_outcomes).split(","):
        if ":" not in piece:
            continue
        key, value = piece.split(":", 1)
        parsed[key.strip()] = int(value)
    return parsed


def plot_timeline_raster(dfs: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Any]:
    events = dfs["tool_timeline_events"]
    time_to_gold = dfs["time_to_gold_file"]
    disagreement = dfs["representation_disagreement_cases"]

    requested_cases = [
        ("Trace-only", "astropy__astropy-15082"),
        ("Trace-only", "django__django-1891"),
        ("Digest-only", "django__django-33871"),
        ("Digest-only", "django__django-34570"),
        ("Action-only", "scikit-learn__scikit-learn-15093"),
        ("Action-only", "sympy__sympy-19484"),
        ("None-only", "matplotlib__matplotlib-22482"),
        ("All prior", "mwaskom__seaborn-3091"),
    ]
    available_cases = set(time_to_gold["instance_id"]) & set(events["instance_id"])
    selected_cases = [(kind, case) for kind, case in requested_cases if case in available_cases]
    dropped_cases = [case for _, case in requested_cases if case not in available_cases]

    cap_step = 60
    selected_ids = [case for _, case in selected_cases]
    selected_events = events.loc[events["instance_id"].isin(selected_ids)].copy()
    selected_time = time_to_gold.loc[time_to_gold["instance_id"].isin(selected_ids)].copy()
    plotted_events = selected_events.loc[
        selected_events["normalized_command_type"].isin(["search", "file_read", "file_edit", "test", "repro"])
        | (selected_events["tool_error_bool"] == 1)
    ].copy()
    plotted_events["x_plot"] = plotted_events["step_index"].clip(upper=cap_step)

    n_rows = len(selected_cases) * len(CONDITIONS)
    fig_height = max(5.8, n_rows * 0.19 + 1.6)
    fig, ax = plt.subplots(figsize=(7.4, fig_height), constrained_layout=True)

    row_lookup: dict[tuple[str, str], int] = {}
    ytick_positions = []
    ytick_labels = []
    case_centers: dict[str, float] = {}
    row_idx = 0
    for case_kind, case in selected_cases:
        block_ys = []
        for condition in CONDITIONS:
            y = n_rows - 1 - row_idx
            row_lookup[(case, condition)] = y
            block_ys.append(y)
            ax.text(
                -1.1,
                y,
                LABELS[condition],
                ha="right",
                va="center",
                fontsize=6.4,
                color=COLORS[condition],
            )
            row_idx += 1
        case_centers[case] = float(np.mean(block_ys))
        ax.axhline(min(block_ys) - 0.5, color="#E0E0E0", linewidth=0.7)
        ax.text(
            -12.1,
            case_centers[case],
            f"{case_kind}\n{short_instance(case)}",
            ha="right",
            va="center",
            fontsize=6.8,
        )

    event_styles = [
        ("search", "search", "o", "#E69F00", 12),
        ("file_read", "file read", "|", "#56B4E9", 35),
        ("file_edit", "edit", "s", "#D55E00", 18),
        ("test", "test/repro", "^", "#009E73", 18),
        ("repro", "test/repro", "^", "#009E73", 18),
    ]
    for command_type, _label, marker, color, size in event_styles:
        mask = plotted_events["normalized_command_type"] == command_type
        if not mask.any():
            continue
        ys = [
            row_lookup.get((case, condition), np.nan)
            for case, condition in zip(
                plotted_events.loc[mask, "instance_id"],
                plotted_events.loc[mask, "condition"],
            )
        ]
        valid = ~pd.isna(pd.Series(ys)).to_numpy()
        data = plotted_events.loc[mask].iloc[np.where(valid)[0]]
        ax.scatter(
            data["x_plot"],
            np.array(ys)[valid],
            marker=marker,
            s=size,
            color=color,
            alpha=0.72,
            linewidths=0.8,
            zorder=2,
        )

    error_mask = plotted_events["tool_error_bool"] == 1
    if error_mask.any():
        ys = [
            row_lookup.get((case, condition), np.nan)
            for case, condition in zip(
                plotted_events.loc[error_mask, "instance_id"],
                plotted_events.loc[error_mask, "condition"],
            )
        ]
        valid = ~pd.isna(pd.Series(ys)).to_numpy()
        data = plotted_events.loc[error_mask].iloc[np.where(valid)[0]]
        ax.scatter(
            data["x_plot"],
            np.array(ys)[valid],
            marker="x",
            s=18,
            color="#000000",
            alpha=0.75,
            linewidths=0.8,
            zorder=4,
        )

    time_lookup = selected_time.set_index(["instance_id", "condition"])
    timeline_summary: dict[str, Any] = {
        "selected_cases": [],
        "cap_step": cap_step,
        "event_rows_available_for_selected_cases": int(len(selected_events)),
        "event_marks_plotted": int(len(plotted_events)),
        "dropped_cases": dropped_cases,
    }
    for case_kind, case in selected_cases:
        case_row = disagreement.loc[disagreement["instance_id"] == case]
        case_summary = {
            "case_type": case_kind,
            "instance_id": case,
            "condition_outcomes": str(case_row["condition_outcomes"].iloc[0]) if len(case_row) else None,
            "runs": {},
        }
        for condition in CONDITIONS:
            key = (case, condition)
            if key not in time_lookup.index or key not in row_lookup:
                continue
            row = time_lookup.loc[key]
            y = row_lookup[key]
            read_step = row["first_gold_file_read_step"]
            edit_step = row["first_gold_file_edit_step"]
            if pd.notna(read_step):
                marker = ">" if float(read_step) > cap_step else "*"
                ax.scatter(
                    [min(float(read_step), cap_step)],
                    [y],
                    marker=marker,
                    s=55,
                    facecolor="white",
                    edgecolor="#111111",
                    linewidth=0.8,
                    zorder=5,
                )
            if pd.notna(edit_step):
                marker = ">" if float(edit_step) > cap_step else "D"
                ax.scatter(
                    [min(float(edit_step), cap_step)],
                    [y],
                    marker=marker,
                    s=40,
                    facecolor="#F0E442",
                    edgecolor="#111111",
                    linewidth=0.7,
                    zorder=5,
                )
            resolved = int(row["resolved"])
            ax.scatter(
                [cap_step + 3.0],
                [y],
                marker="o",
                s=30,
                facecolor=COLORS[condition] if resolved else "white",
                edgecolor=COLORS[condition],
                linewidth=1.1,
                zorder=5,
            )
            case_summary["runs"][condition] = {
                "resolved": resolved,
                "first_gold_file_read_step": None if pd.isna(read_step) else float(read_step),
                "first_gold_file_edit_step": None if pd.isna(edit_step) else float(edit_step),
                "num_tool_calls": int(row["num_tool_calls"]),
                "empty_patch": int(row["empty_patch"]),
                "ran_test_bool": int(row["ran_test_bool"]),
            }
        timeline_summary["selected_cases"].append(case_summary)

    ax.axvline(cap_step, color="#999999", linewidth=0.8, linestyle=":")
    ax.text(cap_step, n_rows + 0.15, "cap", ha="center", va="bottom", fontsize=6.5, color="#666666")
    ax.set_xlim(-13.5, cap_step + 5.5)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, cap_step + 1, 10))
    ax.set_xlabel("Tool step (events after 60 clipped)")
    ax.set_title("Disagreement-case timeline raster")
    ax.grid(axis="x", color="#E6E6E6")
    ax.grid(axis="y", visible=False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#E69F00", markeredgecolor="#E69F00", markersize=4, label="search"),
        Line2D([0], [0], marker="|", color="#56B4E9", markersize=8, linestyle="None", label="file read"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="white", markeredgecolor="#111111", markersize=7, label="target read"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#D55E00", markeredgecolor="#D55E00", markersize=5, label="edit"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#F0E442", markeredgecolor="#111111", markersize=5, label="target edit"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#009E73", markeredgecolor="#009E73", markersize=5, label="test/repro"),
        Line2D([0], [0], marker="x", color="#000000", markersize=5, linestyle="None", label="error"),
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="white", markersize=5, linestyle="None", label="final unresolved"),
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="#333333", markersize=5, linestyle="None", label="final resolved"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=5,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    files = save_figure(fig, out_dir, "fig_disagreement_timeline_raster")
    return {
        "figure": "Figure 5",
        **files,
        "source_csvs": [
            "representation_disagreement_cases.csv",
            "qualitative_case_index.csv",
            "tool_timeline_events.csv",
            "time_to_gold_file.csv",
        ],
        "exact_counts_used": timeline_summary,
        "caption_takeaway": (
            "Different memory representations route the same target through different evidence and "
            "action paths, explaining why aggregate solve counts hide complementarity."
        ),
        "caveats": [
            "Selected qualitative disagreement cases are illustrative, not an additional aggregate test.",
            "Tool steps are capped at 60 for compactness; target read/edit markers beyond the cap are clipped at the cap.",
            "Transcript snippets are intentionally omitted from the main figure.",
        ],
    }


def plot_evidence_action_heatmap(dfs: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Any]:
    evidence = dfs["evidence_target_overlap"]
    inventory = dfs["representation_evidence_inventory"]
    time_to_gold = dfs["time_to_gold_file"]
    require(
        inventory.groupby("condition")["instance_id"].nunique().to_dict()
        == {c: N_TARGETS for c in PRIOR_CONDITIONS},
        "representation_evidence_inventory.csv does not have 95 targets per prior representation",
    )
    action = time_to_gold.loc[time_to_gold["condition"].isin(PRIOR_CONDITIONS)][
        [
            "instance_id",
            "condition",
            "first_gold_file_read_step",
            "edited_target_gold_file_bool",
            "ran_test_bool",
            "resolved",
        ]
    ].copy()
    action["agent_reads_target_gold_file"] = action["first_gold_file_read_step"].notna().astype(int)
    merged = evidence.merge(action, on=["instance_id", "condition"], how="inner")
    require(len(merged) == 285, "Evidence/action merge did not produce 285 rows")

    columns = [
        ("evidence_mentions_target_gold_file_bool", "evidence mentions\ntarget gold file"),
        ("evidence_mentions_target_gold_dir_bool", "evidence mentions\ntarget gold dir"),
        ("evidence_contains_relevant_test_command_bool", "evidence contains\nrelevant test command"),
        (
            "evidence_contains_prior_inspected_target_gold_file_bool",
            "evidence contains prior\ninspected gold file",
        ),
        ("agent_reads_target_gold_file", "agent reads\ntarget gold file"),
        ("edited_target_gold_file_bool", "agent edits\ntarget gold file"),
        ("ran_test_bool", "agent runs\ntests"),
        ("resolved", "resolved"),
    ]
    row_conditions = PRIOR_CONDITIONS
    counts = np.zeros((len(row_conditions), len(columns)), dtype=int)
    ns: dict[str, int] = {}
    for i, condition in enumerate(row_conditions):
        rows = merged.loc[merged["condition"] == condition]
        ns[condition] = int(len(rows))
        for j, (column, _label) in enumerate(columns):
            counts[i, j] = int(rows[column].sum())
    rates = counts / np.array([ns[c] for c in row_conditions])[:, None]

    fig, ax = plt.subplots(figsize=(7.3, 2.85), constrained_layout=True)
    image = ax.imshow(rates, aspect="auto", vmin=0, vmax=1, cmap="cividis")
    ax.set_yticks(np.arange(len(row_conditions)))
    ax.set_yticklabels([LABELS[c] for c in row_conditions])
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([label for _, label in columns], rotation=35, ha="right")
    ax.set_title("Evidence-to-action rates by representation")
    for i, condition in enumerate(row_conditions):
        for j, _ in enumerate(columns):
            rate = rates[i, j]
            color = "white" if rate > 0.62 else "black"
            ax.text(
                j,
                i,
                f"{counts[i, j]}/{ns[condition]}\n{rate * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=6.2,
                color=color,
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("Rate")

    files = save_figure(fig, out_dir, "appendix_fig_evidence_action_heatmap")
    count_table = {
        condition: {
            label.replace("\n", " "): {"count": int(counts[i, j]), "n": ns[condition], "rate": float(rates[i, j])}
            for j, (_column, label) in enumerate(columns)
        }
        for i, condition in enumerate(row_conditions)
    }
    return {
        "figure": "Appendix Figure 6",
        **files,
        "source_csvs": [
            "evidence_target_overlap.csv",
            "representation_evidence_inventory.csv",
            "time_to_gold_file.csv",
        ],
        "exact_counts_used": count_table,
        "caption_takeaway": (
            "Representation-exposed evidence is common, but evidence presence alone is not sufficient "
            "for success; the agent must act on it."
        ),
        "caveats": [
            "Evidence-to-action rates are post hoc and not causal.",
            "Rows exclude the no-prior-context condition because the evidence columns concern prior representations.",
        ],
    }


def write_caption_snippets(out_dir: Path) -> None:
    captions = r"""
% Auto-generated by figures/make_figures.py.

\begin{figure*}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_paired_effects.pdf}
  \caption{\textbf{Paired effects versus no prior context.} Points show paired percentage-point differences against None with paired bootstrap 95\% intervals. Resolution gains are descriptive and paired-success McNemar tests are not Holm-significant, while non-empty patch-attempt effects are large and Holm-significant. These comparisons do not imply causal mediation.}
  \label{fig:paired-effects}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_solve_complementarity_oracle.pdf}
  \caption{\textbf{Solve complementarity and retrospective oracle headroom.} Trace has the highest fixed prior-context solve count, but different prior representations solve different targets. A retrospective prior-context oracle reaches 28/95, which is +9 over the best fixed prior representation, while the all-condition retrospective oracle reaches 29/95. The oracle is retrospective and not a deployable method.}
  \label{fig:solve-complementarity-oracle}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_localization_map.pdf}
  \caption{\textbf{Localization map.} Prior-context gains concentrate when prior and target trajectories are local in file or directory space. The ECDF denominator is all 95 targets per condition; runs with missing first target-gold-file reads are counted as never reached. Prior-context agents reach target-gold files earlier and use fewer mean search commands, consistent with localization/procedural transfer rather than proving a causal mechanism.}
  \label{fig:localization-map}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_patch_action_funnel.pdf}
  \caption{\textbf{Patch/action funnel.} Prior-context conditions primarily reduce empty-patch behavior and increase target-relevant actions before the narrower final-resolution endpoint. Because running tests is not strictly sequential after editing, this is an action funnel rather than a causal pipeline.}
  \label{fig:patch-action-funnel}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_disagreement_timeline_raster.pdf}
  \caption{\textbf{Disagreement-case timeline raster.} Selected targets where representations disagree show that different memory renderings can route the agent through different evidence and action paths on the same target. Tool steps are capped at 60 for compactness, and snippets are omitted from the main figure.}
  \label{fig:disagreement-timeline-raster}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/appendix_fig_evidence_action_heatmap.pdf}
  \caption{\textbf{Evidence-to-action heatmap.} Representation-exposed evidence is often present, but evidence presence is not sufficient for success; the agent must read, edit, test, and ultimately resolve. This post hoc analysis is descriptive and not causal.}
  \label{fig:evidence-action-heatmap}
\end{figure*}
""".strip()
    (out_dir / "caption_snippets.tex").write_text(captions + "\n", encoding="utf-8")


def write_readme(out_dir: Path, manifest: dict[str, Any]) -> None:
    figure_lines = []
    for entry in manifest["figures"]:
        figure_lines.append(
            f"- {entry['filename']} / {entry['png_filename']}: {entry['caption_takeaway']}"
        )
    text = f"""# Generated Figures

Run:

```bash
python figures/make_figures.py --data-dir data --out-dir figures
```

The script validates the expected 95-target analysis counts before plotting:
condition solve counts, non-empty patch counts, solve-pattern counts, oracle
counts, exact McNemar/bootstrap columns, and the 95 x 4 time-to-gold-file
structure. It fails loudly if any required CSV is missing or any headline count
drifts.

The plots use pandas and matplotlib only. PDFs are vector outputs intended for
LaTeX inclusion; PNGs are high-resolution previews. The palette is colorblind
safe and avoids decorative effects.

## Outputs

{chr(10).join(figure_lines)}

## Scientific Notes

- This is not a leaderboard figure set.
- The retrospective oracle is labeled as retrospective and not deployable.
- Mechanism figures use the wording "consistent with localization/procedural transfer" and do not claim causality.
- The action funnel is not a causal pipeline because test-running is not strictly sequential after editing.
- The timeline raster uses structured event data and omits transcript snippets from the main figure.

## Remote Transcript Provenance

Selected run directories under `/mnt/data/shashwat/openhands-adp-memory/data/contextbench_phase2/execution_full_qwen36_65k_fix1/runs` were checked read-only for the disagreement cases. The small local provenance file `remote_artifact_check.tsv`, when present, records artifact filenames and sizes. Raw transcript text is not embedded in the figures.

## Validation Summary

```json
{json.dumps(clean_json(manifest['validation']), indent=2, sort_keys=True)}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def print_validation_report(validation: dict[str, Any], figures: list[dict[str, Any]]) -> None:
    print("Validation report")
    print("=================")
    print("Condition solve counts:")
    for condition in CONDITIONS:
        counts = validation["condition_counts"][condition]
        print(
            f"  {condition}: resolved {counts['resolved_count']}/95, "
            f"non-empty patches {counts['non_empty_patch_count']}/95"
        )
    print("Patch-attempt counts:")
    for condition in CONDITIONS:
        counts = validation["patch_attempt_counts"][condition]
        print(
            f"  {condition}: non-empty {counts['non_empty_patch_count']}/95, "
            f"empty {counts['empty_patch_count']}/95"
        )
    print(
        f"Oracle counts: oracle_prior={validation['oracle_prior']}/95, "
        f"oracle_all={validation['oracle_all']}/95, "
        f"headroom={validation['oracle_headroom_over_best_fixed_prior']}"
    )
    paired = next(fig for fig in figures if fig["figure"] == "Figure 1")
    print("Paired effect rows used:")
    for metric, rows in paired["exact_counts_used"].items():
        for row in rows:
            print(
                f"  {metric} {row['comparison']}: "
                f"diff={row['rate_diff_percentage_points']:.1f} p.p., "
                f"CI=[{row['ci_low_percentage_points']:.1f}, {row['ci_high_percentage_points']:.1f}], "
                f"{p_label('p', row['mcnemar_exact_p'])}, {p_label('Holm', row['holm_adjusted_p'])}"
            )
    timeline = next(fig for fig in figures if fig["figure"] == "Figure 5")
    exact = timeline["exact_counts_used"]
    print(
        "Timeline raster rows: "
        f"{exact['event_rows_available_for_selected_cases']} selected event rows, "
        f"{exact['event_marks_plotted']} plotted event marks"
    )
    print("Localization bucket target counts:")
    for bucket, count in validation["localization_bucket_counts"].items():
        print(f"  {bucket}: {count}")
    dropped = exact["dropped_cases"]
    if dropped:
        print(f"Missing transcripts or dropped timeline cases: {', '.join(dropped)}")
    else:
        print("Missing transcripts or dropped timeline cases: none in structured event data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dfs = load_data(args.data_dir)
    validation = validate_counts(dfs)

    figures = [
        plot_paired_effects(dfs, args.out_dir),
        plot_solve_complementarity_oracle(dfs, args.out_dir),
        plot_localization_map(dfs, args.out_dir),
        plot_patch_action_funnel(dfs, args.out_dir),
        plot_timeline_raster(dfs, args.out_dir),
        plot_evidence_action_heatmap(dfs, args.out_dir),
    ]
    manifest = {
        "data_dir": str(args.data_dir),
        "validation": validation,
        "remote_transcript_note": (
            "Selected remote run artifact directories under /mnt/data/shashwat were checked "
            "read-only; the timeline figure uses structured event CSVs and no transcript snippets."
        ),
        "figures": figures,
    }
    (args.out_dir / "figure_manifest.json").write_text(
        json.dumps(clean_json(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_caption_snippets(args.out_dir)
    write_readme(args.out_dir, manifest)
    print_validation_report(validation, figures)


if __name__ == "__main__":
    main()
