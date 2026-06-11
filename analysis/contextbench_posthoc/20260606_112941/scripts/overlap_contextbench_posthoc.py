#!/usr/bin/env python3
"""Overlap/localization diagnostics for ContextBench posthoc analysis."""

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
EXCLUDED_TARGET = "django__django-28147"


OVERLAP_COLUMNS = [
    "instance_id",
    "repo",
    "prior_instance_id",
    "target_instance_id",
    "prior_gold_files",
    "target_gold_files",
    "prior_gold_file_count",
    "target_gold_file_count",
    "prior_gold_target_gold_file_intersection",
    "prior_gold_target_gold_file_intersection_count",
    "prior_gold_target_gold_jaccard",
    "prior_gold_target_gold_same_file_bool",
    "prior_gold_target_gold_same_dir_bool",
    "prior_trajectory_inspected_files",
    "prior_trajectory_edited_files",
    "prior_trajectory_test_files",
    "prior_trajectory_source_files",
    "prior_inspected_target_gold_intersection",
    "prior_inspected_target_gold_intersection_count",
    "prior_inspected_target_gold_jaccard",
    "prior_inspected_target_gold_same_file_bool",
    "prior_inspected_target_gold_same_dir_bool",
    "prior_edited_target_gold_intersection",
    "prior_edited_target_gold_intersection_count",
    "prior_edited_target_gold_jaccard",
    "prior_edited_target_gold_same_file_bool",
    "prior_edited_target_gold_same_dir_bool",
    "raw_resolved",
    "adp_resolved",
    "memory_resolved",
    "no_memory_resolved",
    "overlap_bucket",
    "localization_bucket",
]


BUCKET_COLUMNS = [
    "bucket_type",
    "bucket",
    "condition",
    "n_targets",
    "resolved_count",
    "success_rate",
    "non_empty_patch_count",
    "patch_attempt_rate",
    "mean_runtime",
    "mean_total_tokens",
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def normalize_path(path: str) -> str:
    path = path.replace("\\", "/").strip()
    path = re.sub(r"^\./", "", path)
    for marker in ["/testbed/", "testbed/"]:
        if marker in path:
            path = path.split(marker, 1)[1]
    path = re.sub(r"^swebench_[^/]+/", "", path)
    parts = [p for p in path.split("/") if p not in {"", "."}]
    if parts and "__" in parts[0]:
        parts = parts[1:]
    if parts and parts[0] in {"output", "tmp"}:
        return "/".join(parts)
    return "/".join(parts)


def is_test_file(path: str) -> bool:
    low = path.lower()
    return "/test/" in low or "/tests/" in low or low.startswith("test_") or "/test_" in low or low.endswith("_test.py")


def split_set(value: str) -> set[str]:
    if not value:
        return set()
    return {normalize_path(x) for x in value.split(";") if normalize_path(x)}


def set_str(paths: Iterable[str]) -> str:
    return ";".join(sorted(p for p in paths if p))


def dirs(paths: Iterable[str]) -> set[str]:
    out = set()
    for p in paths:
        parent = str(Path(p).parent).replace("\\", "/")
        if parent and parent != ".":
            out.add(parent)
    return out


def jaccard(a: set[str], b: set[str]) -> str:
    if not a or not b:
        return ""
    return f"{len(a & b) / len(a | b):.6g}"


def same_dir(a: set[str], b: set[str]) -> str:
    if not a or not b:
        return ""
    return "1" if dirs(a) & dirs(b) else "0"


def parse_diff_files(diff_text: str) -> set[str]:
    files: set[str] = set()
    for line in diff_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        candidates = []
        for p in parts[2:4]:
            if p.startswith("a/") or p.startswith("b/"):
                p = p[2:]
            if p != "/dev/null":
                candidates.append(normalize_path(p))
        for p in reversed(candidates):
            if p:
                files.add(p)
                break
    return files


def build_official_patch_index(repo_root: Path) -> tuple[dict[str, set[str]], list[str]]:
    """Best-effort official gold patch index.

    The local phase-2 manifests currently do not carry gold patch fields. This
    function scans small manifest/dataset-style JSON/JSONL files but deliberately
    avoids generated prediction/model_patch artifacts and raw prior trajectories.
    """
    patch_index: dict[str, set[str]] = {}
    sources: list[str] = []
    scan_roots = [
        repo_root / "data/contextbench_phase2",
        repo_root / "data/contextbench_dataset",
        repo_root / "data/nebius_experiment",
    ]
    patch_field_names = {"patch", "gold_patch", "test_patch"}
    for root in scan_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {"workspace", ".git", "__pycache__"}]
            for name in filenames:
                low = name.lower()
                if not (low.endswith(".json") or low.endswith(".jsonl")):
                    continue
                if "prediction" in low or "_preds" in low or "model_patch" in low:
                    continue
                path = Path(dirpath) / name
                try:
                    rows = read_jsonl(path) if low.endswith(".jsonl") else [json.loads(path.read_text(encoding="utf-8", errors="replace"))]
                except Exception:
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    instance = row.get("instance_id") or row.get("target_instance_id") or row.get("related_instance_id")
                    if not instance:
                        continue
                    for field in patch_field_names:
                        value = row.get(field)
                        if isinstance(value, str) and "diff --git " in value:
                            patch_index.setdefault(str(instance), set()).update(parse_diff_files(value))
                            sources.append(str(path))
    return patch_index, sorted(set(sources))


def build_prior_trajectory_index(repo_root: Path) -> dict[str, dict[str, set[str]]]:
    """Use deterministic memory JSONL as a structured file-activity source."""
    path = repo_root / "data/contextbench_phase1/stripped/generated_memory_from_stripped_adp.jsonl"
    out: dict[str, dict[str, set[str]]] = {}
    for row in read_jsonl(path):
        tid = str(row.get("source_trajectory_id", ""))
        if not tid:
            continue
        inspected: set[str] = set()
        edited: set[str] = set()
        files = row.get("files") if isinstance(row.get("files"), dict) else {}
        for item in files.get("inspected", []) if isinstance(files, dict) else []:
            if isinstance(item, dict) and item.get("path"):
                inspected.add(normalize_path(str(item["path"])))
        for item in files.get("edited", []) if isinstance(files, dict) else []:
            if isinstance(item, dict) and item.get("path"):
                edited.add(normalize_path(str(item["path"])))
        # Edits are duplicated in a more semantic structure; include them too.
        edits = row.get("edits") if isinstance(row.get("edits"), dict) else {}
        for key in ["source_edits", "test_edits", "other_edits"]:
            for item in edits.get(key, []) if isinstance(edits, dict) else []:
                if isinstance(item, dict) and item.get("path"):
                    edited.add(normalize_path(str(item["path"])))
        test_files = {p for p in inspected | edited if is_test_file(p)}
        source_files = {p for p in inspected | edited if not is_test_file(p)}
        out[tid] = {
            "inspected": inspected,
            "edited": edited,
            "test_files": test_files,
            "source_files": source_files,
        }
    return out


def load_prompt_index(repo_root: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(repo_root / "data/contextbench_phase2/prompt_manifest.jsonl")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = str(row.get("target_instance_id", ""))
        if target and target not in out:
            out[target] = row
    return out


def load_pair_index(repo_root: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(repo_root / "data/contextbench_phase2/pair_manifest.jsonl")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = str(row.get("related_instance_id", ""))
        if target:
            out[target] = row
    return out


def gold_bucket(prior_gold: set[str], target_gold: set[str]) -> str:
    if not prior_gold or not target_gold:
        return "unknown_gold_overlap"
    if prior_gold & target_gold:
        return "same_file_overlap"
    if dirs(prior_gold) & dirs(target_gold):
        return "same_directory_only"
    return "no_gold_file_overlap"


def localization_bucket(inspected: set[str], edited: set[str], target_gold: set[str]) -> str:
    if not target_gold:
        return "unknown"
    if edited & target_gold:
        return "prior_trajectory_edited_target_gold_file"
    if inspected & target_gold:
        return "prior_trajectory_inspected_target_gold_file"
    if dirs(inspected | edited) & dirs(target_gold):
        return "prior_trajectory_same_directory_as_target_gold"
    return "prior_trajectory_never_touched_target_gold_area"


def build_overlap_rows(repo_root: Path, paired_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    prompt_index = load_prompt_index(repo_root)
    pair_index = load_pair_index(repo_root)
    gold_index, gold_sources = build_official_patch_index(repo_root)
    prior_index = build_prior_trajectory_index(repo_root)
    rows: list[dict[str, Any]] = []
    for paired in paired_rows:
        target = paired.get("instance_id", "")
        if not target or target == EXCLUDED_TARGET:
            continue
        prompt = prompt_index.get(target, {})
        pair = pair_index.get(target, {})
        repo = paired.get("repo") or prompt.get("target_repo") or pair.get("repo") or ""
        prior = paired.get("prior_instance_id") or prompt.get("prior_instance_id") or pair.get("experience_instance_id") or ""
        tid = str(prompt.get("prior_trajectory_id") or pair.get("prior_trajectory_id") or "")
        prior_files = prior_index.get(tid, {})
        inspected = set(prior_files.get("inspected", set()))
        edited = set(prior_files.get("edited", set()))
        test_files = set(prior_files.get("test_files", set()))
        source_files = set(prior_files.get("source_files", set()))
        prior_gold = gold_index.get(str(prior), set())
        target_gold = gold_index.get(str(target), set())
        gold_inter = prior_gold & target_gold
        inspected_inter = inspected & target_gold
        edited_inter = edited & target_gold
        rows.append(
            {
                "instance_id": target,
                "repo": repo,
                "prior_instance_id": prior,
                "target_instance_id": target,
                "prior_gold_files": set_str(prior_gold),
                "target_gold_files": set_str(target_gold),
                "prior_gold_file_count": len(prior_gold) if prior_gold else "",
                "target_gold_file_count": len(target_gold) if target_gold else "",
                "prior_gold_target_gold_file_intersection": set_str(gold_inter),
                "prior_gold_target_gold_file_intersection_count": len(gold_inter) if prior_gold and target_gold else "",
                "prior_gold_target_gold_jaccard": jaccard(prior_gold, target_gold),
                "prior_gold_target_gold_same_file_bool": "1" if gold_inter else ("0" if prior_gold and target_gold else ""),
                "prior_gold_target_gold_same_dir_bool": same_dir(prior_gold, target_gold),
                "prior_trajectory_inspected_files": set_str(inspected),
                "prior_trajectory_edited_files": set_str(edited),
                "prior_trajectory_test_files": set_str(test_files),
                "prior_trajectory_source_files": set_str(source_files),
                "prior_inspected_target_gold_intersection": set_str(inspected_inter),
                "prior_inspected_target_gold_intersection_count": len(inspected_inter) if target_gold else "",
                "prior_inspected_target_gold_jaccard": jaccard(inspected, target_gold),
                "prior_inspected_target_gold_same_file_bool": "1" if inspected_inter else ("0" if target_gold else ""),
                "prior_inspected_target_gold_same_dir_bool": same_dir(inspected, target_gold),
                "prior_edited_target_gold_intersection": set_str(edited_inter),
                "prior_edited_target_gold_intersection_count": len(edited_inter) if target_gold else "",
                "prior_edited_target_gold_jaccard": jaccard(edited, target_gold),
                "prior_edited_target_gold_same_file_bool": "1" if edited_inter else ("0" if target_gold else ""),
                "prior_edited_target_gold_same_dir_bool": same_dir(edited, target_gold),
                "raw_resolved": paired.get("resolved_raw", ""),
                "adp_resolved": paired.get("resolved_adp", ""),
                "memory_resolved": paired.get("resolved_memory", ""),
                "no_memory_resolved": paired.get("resolved_no_memory", ""),
                "overlap_bucket": gold_bucket(prior_gold, target_gold),
                "localization_bucket": localization_bucket(inspected, edited, target_gold),
            }
        )
    return rows, gold_sources


def numeric_mean(rows: list[dict[str, str]], key: str) -> str:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, "")))
        except (TypeError, ValueError):
            pass
    return "" if not vals else f"{statistics.fmean(vals):.6g}"


def bucket_summary(overlap_rows: list[dict[str, Any]], paired_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    paired = {r.get("instance_id", ""): r for r in paired_rows}
    out: list[dict[str, Any]] = []
    for bucket_type, col in [("overlap_bucket", "overlap_bucket"), ("localization_bucket", "localization_bucket")]:
        buckets = sorted(set(str(r.get(col, "")) for r in overlap_rows if r.get(col, "")))
        for bucket in buckets:
            ids = [str(r["instance_id"]) for r in overlap_rows if r.get(col) == bucket]
            group = [paired[i] for i in ids if i in paired]
            for cond in CONDITIONS:
                res_vals = [r.get(f"resolved_{cond}", "") for r in group]
                attempt_vals = [r.get(f"non_empty_patch_{cond}", "") for r in group]
                res_complete = all(v in {"0", "1"} for v in res_vals) and bool(res_vals)
                attempt_complete = all(v in {"0", "1"} for v in attempt_vals) and bool(attempt_vals)
                out.append(
                    {
                        "bucket_type": bucket_type,
                        "bucket": bucket,
                        "condition": cond,
                        "n_targets": len(group),
                        "resolved_count": sum(int(v) for v in res_vals) if res_complete else "",
                        "success_rate": f"{sum(int(v) for v in res_vals) / len(group):.6g}" if res_complete and group else "",
                        "non_empty_patch_count": sum(int(v) for v in attempt_vals) if attempt_complete else "",
                        "patch_attempt_rate": f"{sum(int(v) for v in attempt_vals) / len(group):.6g}" if attempt_complete and group else "",
                        "mean_runtime": numeric_mean(group, f"wall_seconds_{cond}"),
                        "mean_total_tokens": numeric_mean(group, f"total_tokens_{cond}"),
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


def write_report(out_dir: Path, overlap_rows: list[dict[str, Any]], bucket_rows: list[dict[str, Any]], gold_sources: list[str]) -> None:
    bucket_counts = defaultdict(int)
    for row in overlap_rows:
        bucket_counts[row["overlap_bucket"]] += 1
    loc_counts = defaultdict(int)
    for row in overlap_rows:
        loc_counts[row["localization_bucket"]] += 1
    lines = [
        "# Overlap Results",
        "",
        "## Official Gold Patch Availability",
    ]
    if gold_sources:
        lines.append("Official patch-like fields were found in:")
        lines.extend(f"- `{s}`" for s in gold_sources)
    else:
        lines.append(
            "No official `patch`, `gold_patch`, or `test_patch` fields were found in the local phase-2/dataset manifests scanned. "
            "Generated model patches were not used as gold patches."
        )
    lines.extend(
        [
            "",
            "## Gold Overlap Buckets",
            markdown_table([{"bucket": k, "count": v} for k, v in sorted(bucket_counts.items())], ["bucket", "count"]),
            "",
            "## Prior Trajectory Localization Buckets",
            markdown_table([{"bucket": k, "count": v} for k, v in sorted(loc_counts.items())], ["bucket", "count"]),
            "",
            "## Bucket Summary",
            markdown_table(bucket_rows, BUCKET_COLUMNS),
            "",
            "Interpretation should remain descriptive until official target/prior gold patches are available. "
            "Current local output can support prior trajectory file-activity summaries but cannot separate broad transfer from path leakage/localization with gold-file evidence.",
        ]
    )
    write_text(out_dir / "reports/overlap_results.md", "\n".join(lines) + "\n")
    paper_path = out_dir / "reports/paper_ready_tables.md"
    existing = paper_path.read_text(encoding="utf-8") if paper_path.exists() else "# Paper-Ready Tables\n"
    if "## Overlap Bucket Results" not in existing:
        addition = [
            "",
            "## Overlap Bucket Results",
            markdown_table(bucket_rows, BUCKET_COLUMNS),
        ]
        write_text(paper_path, existing.rstrip() + "\n" + "\n".join(addition) + "\n")
    # Create placeholder/real bucket figure.
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        labels = list(sorted(bucket_counts))
        counts = [bucket_counts[l] for l in labels]
        plt.figure(figsize=(7, 3.5))
        if labels:
            plt.bar(labels, counts)
            plt.xticks(rotation=20, ha="right")
            plt.ylabel("targets")
        else:
            plt.text(0.5, 0.5, "Data unavailable", ha="center", va="center")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(fig_dir / "overlap_bucket_success.png", dpi=140)
        plt.close()
    except Exception as exc:  # noqa: BLE001
        (out_dir / "figures/overlap_bucket_success.png.txt").write_text(f"Figure unavailable: {exc}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    paired_rows = read_csv(out_dir / "data/paired_results.csv")
    eprint(f"[overlap] loaded {len(paired_rows)} paired rows")
    overlap_rows, gold_sources = build_overlap_rows(repo_root, paired_rows)
    bucket_rows = bucket_summary(overlap_rows, paired_rows)
    write_csv(out_dir / "data/overlap_features.csv", overlap_rows, OVERLAP_COLUMNS)
    write_csv(out_dir / "data/overlap_bucket_summary.csv", bucket_rows, BUCKET_COLUMNS)
    write_report(out_dir, overlap_rows, bucket_rows, gold_sources)
    if not gold_sources:
        eprint("[overlap] official gold patches unavailable locally; gold overlap marked unknown")
        return 2
    eprint("[overlap] completed overlap analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
