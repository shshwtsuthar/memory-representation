#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


CONDITIONS = ["no_memory", "raw", "adp", "memory"]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pick_file(files: list[str], *, must: list[str], avoid: list[str] = ()) -> str:
    candidates = []
    for f in files:
        name = Path(f).name
        low = name.lower()
        if not low.endswith(".parquet"):
            continue
        if all(m.lower() in low for m in must) and not any(a.lower() in low for a in avoid):
            candidates.append(f)

    if not candidates:
        raise RuntimeError(f"No parquet matched must={must}, avoid={avoid}. Files: {files}")

    # Prefer shorter/simple names when multiple match.
    return sorted(candidates, key=lambda x: (len(Path(x).name), x))[0]


def download_contextbench_files(out_dir: Path, lite: bool) -> dict[str, Path]:
    repo_id = "jiayuanz3/SWEContextBench"
    files = list_repo_files(repo_id, repo_type="dataset")

    if lite:
        exp_file = pick_file(files, must=["experience", "lite"])
        rel_file = pick_file(files, must=["related", "lite"])
    else:
        exp_file = pick_file(files, must=["experience"], avoid=["lite"])
        rel_file = pick_file(files, must=["related"], avoid=["lite", "relationship"])

    relationship_file = pick_file(files, must=["relationship"])

    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for key, filename in {
        "experience": exp_file,
        "related": rel_file,
        "relationship": relationship_file,
    }.items():
        paths[key] = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=out_dir,
            )
        )

    return paths


def build_adp_index(adp_jsonl: Path) -> dict[str, dict]:
    """
    Returns:
      experience_instance_id -> trajectory metadata
    """
    out = {}

    for rec in read_jsonl(adp_jsonl):
        traj_id = rec.get("id")
        details = rec.get("details") or {}
        issue_meta = details.get("issue_metadata") or {}

        instance_id = issue_meta.get("instance_id")
        repo = issue_meta.get("repo")
        base_commit = issue_meta.get("base_commit")
        source_file = details.get("source_file")

        if not instance_id:
            continue

        if instance_id in out:
            raise RuntimeError(f"Duplicate experience instance_id in ADP: {instance_id}")

        out[instance_id] = {
            "prior_instance_id": instance_id,
            "prior_trajectory_id": traj_id,
            "prior_repo": repo,
            "prior_base_commit": base_commit,
            "prior_source_file": source_file,
        }

    return out


def index_injection_files(directory: Path) -> dict[str, Path]:
    """
    Index by trajectory id or instance id based on filename/text.

    Expected current filenames usually contain the trajectory id.
    This fallback also scans the first part of text files for ids.
    """
    out = {}

    if not directory.exists():
        raise RuntimeError(f"Missing injection directory: {directory}")

    for p in sorted(directory.glob("*.txt")):
        name = p.name

        # Common case: 000001_<trajectory_id>.txt
        stem_parts = p.stem.split("_")
        for part in stem_parts:
            if len(part) >= 6:
                out.setdefault(part, p)

        # Fallback: scan small prefix for explicit ids.
        try:
            text = p.read_text(encoding="utf-8", errors="replace")[:12000]
        except Exception:
            continue

        for marker in ["trajectory_id:", "source_trajectory_id:", "instance_id:"]:
            for line in text.splitlines():
                low = line.lower()
                if marker in low:
                    value = line.split(":", 1)[-1].strip()
                    if value:
                        out.setdefault(value, p)

    return out


def docker_tag_for_instance(instance_id: str) -> str:
    # Docker Hub examples use sympy.sympy-21149 for sympy__sympy-21149.
    return instance_id.replace("__", ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-dir", type=Path, default=Path("data/contextbench_phase1"))
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--adp-dir", type=Path, default=None)
    ap.add_argument("--memory-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/contextbench_phase2/run_manifest.jsonl"))
    ap.add_argument("--dataset-cache-dir", type=Path, default=Path("data/contextbench_dataset"))
    ap.add_argument("--lite", action="store_true", help="Use Lite Experience/Related parquet if present.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
    "--pair-mode",
    choices=["one_per_target", "all_pairs"],
    default="one_per_target",
    help=(
        "one_per_target = choose one deterministic prior A for each target B; "
        "all_pairs = keep every unique official A→B pair"
    ),
)
    args = ap.parse_args()

    phase1 = args.phase1_dir

    adp_path = phase1 / "all_trajectories.adp.jsonl"

    def resolve_injection_dir(explicit: Path | None, primary: Path, fallback: Path, label: str) -> Path:
        """
        Resolve injection directories.

        Preferred:
          phase1/raw_transcripts
          phase1/stripped_adp_rendered
          phase1/generated_memory_text_from_stripped_adp

        Also supports the current Phase 1 layout:
          phase1/stripped/raw_transcripts
          phase1/stripped/stripped_adp_rendered
          phase1/stripped/generated_memory_text_from_stripped_adp
        """
        if explicit is not None:
            return explicit
        if primary.exists():
            return primary
        if fallback.exists():
            return fallback
        # Return primary so the later error message remains simple and expected.
        return primary

    raw_dir = resolve_injection_dir(
        args.raw_dir,
        phase1 / "raw_transcripts",
        phase1 / "stripped" / "raw_transcripts",
        "raw",
    )
    adp_dir = resolve_injection_dir(
        args.adp_dir,
        phase1 / "stripped_adp_rendered",
        phase1 / "stripped" / "stripped_adp_rendered",
        "adp",
    )
    memory_dir = resolve_injection_dir(
        args.memory_dir,
        phase1 / "generated_memory_text_from_stripped_adp",
        phase1 / "stripped" / "generated_memory_text_from_stripped_adp",
        "memory",
    )

    print(f"Using raw injection dir:    {raw_dir}")
    print(f"Using ADP injection dir:    {adp_dir}")
    print(f"Using memory injection dir: {memory_dir}")

    if not adp_path.exists():
        raise RuntimeError(f"Missing {adp_path}")

    dataset_paths = download_contextbench_files(args.dataset_cache_dir, lite=args.lite)

    experience_df = pd.read_parquet(dataset_paths["experience"])
    related_df = pd.read_parquet(dataset_paths["related"])
    relationship_df = pd.read_parquet(dataset_paths["relationship"])

    # Restrict full relationship file to the selected split/subset.
    experience_ids = set(experience_df["instance_id"])
    related_ids = set(related_df["instance_id"])

    relationship_df = relationship_df[
        relationship_df["experience_instance_id"].isin(experience_ids)
        & relationship_df["related_instance_id"].isin(related_ids)
    ].copy()

    # Drop exact duplicate A→B rows. The relationship parquet can contain
    # more rows than unique usable pairs after subset filtering.
    relationship_df = (
        relationship_df
        .drop_duplicates(subset=["experience_instance_id", "related_instance_id"])
        .reset_index(drop=True)
    )    

    required_relationship_cols = {"experience_instance_id", "related_instance_id"}
    missing = required_relationship_cols - set(relationship_df.columns)
    if missing:
        raise RuntimeError(f"Relationship parquet missing columns: {missing}")

    related_by_id = {
        row["instance_id"]: row.to_dict()
        for _, row in related_df.iterrows()
    }

    adp_by_experience_id = build_adp_index(adp_path)

    relationship_df = relationship_df[
        relationship_df["experience_instance_id"].isin(set(adp_by_experience_id.keys()))
    ].copy()

    if args.pair_mode == "one_per_target":
        relationship_df = (
            relationship_df
            .sort_values(["related_instance_id", "experience_instance_id"])
            .drop_duplicates(subset=["related_instance_id"], keep="first")
            .reset_index(drop=True)
        )
    else:
        relationship_df = (
            relationship_df
            .sort_values(["related_instance_id", "experience_instance_id"])
            .reset_index(drop=True)
        )

    injection_indexes = {
        "raw": index_injection_files(raw_dir),
        "adp": index_injection_files(adp_dir),
        "memory": index_injection_files(memory_dir),
    }

    rows = []
    pair_rows = []

    for _, rel in relationship_df.iterrows():
        experience_id = rel["experience_instance_id"]
        related_id = rel["related_instance_id"]

        if experience_id not in adp_by_experience_id:
            continue

        if related_id not in related_by_id:
            continue

        prior = adp_by_experience_id[experience_id]
        target = related_by_id[related_id]

        trajectory_id = prior["prior_trajectory_id"]

        # Find injection files for prior trajectory A.
        injection_files = {}
        for condition in ["raw", "adp", "memory"]:
            idx = injection_indexes[condition]

            p = (
                idx.get(str(trajectory_id))
                or idx.get(str(experience_id))
                or None
            )

            if p is None:
                raise RuntimeError(
                    f"Could not find {condition} injection file for "
                    f"experience_id={experience_id}, trajectory_id={trajectory_id}"
                )

            injection_files[condition] = p

        pair_rows.append({
            "experience_instance_id": experience_id,
            "related_instance_id": related_id,
            "prior_trajectory_id": trajectory_id,
            "repo": target.get("repo"),
            "target_base_commit": target.get("base_commit"),
            "target_version": target.get("version"),
        })

        for condition in CONDITIONS:
            if args.pair_mode == "all_pairs":
                run_id = f"{related_id}__from__{experience_id}__{condition}"
            else:
                run_id = f"{related_id}__{condition}"
                
            if condition == "no_memory":
                injection_file = None
            else:
                injection_file = str(injection_files[condition])

            rows.append({
                "run_id": run_id,
                "condition": condition,

                "target_instance_id": related_id,
                "target_repo": target.get("repo"),
                "target_base_commit": target.get("base_commit"),
                "target_version": target.get("version"),
                "target_problem_statement": target.get("problem_statement"),

                # Keep these for evaluator metadata, not for prompt.
                "target_environment_setup_commit": target.get("environment_setup_commit"),
                "target_fail_to_pass": target.get("FAIL_TO_PASS"),
                "target_pass_to_pass": target.get("PASS_TO_PASS"),

                "prior_instance_id": experience_id,
                "prior_trajectory_id": trajectory_id,
                "prior_repo": prior.get("prior_repo"),
                "prior_base_commit": prior.get("prior_base_commit"),
                "prior_source_file": prior.get("prior_source_file"),

                "injection_file": injection_file,

                "sandbox_image": f"jiayuanz3/swecontextbench:{docker_tag_for_instance(related_id)}",

                "output_dir": f"runs/contextbench_qwen3_30b_openhands/{related_id}/{condition}",
                "status": "pending",
            })

        if args.limit is not None and len(pair_rows) >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    pair_out = args.out.with_name("pair_manifest.jsonl")
    with pair_out.open("w", encoding="utf-8") as f:
        for row in pair_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"Pairs: {len(pair_rows)}")
    print(f"Runs:  {len(rows)}")
    print(f"Wrote: {args.out}")
    print(f"Wrote: {pair_out}")

    by_condition = {}
    for row in rows:
        by_condition[row["condition"]] = by_condition.get(row["condition"], 0) + 1
    print("By condition:", by_condition)


if __name__ == "__main__":
    main()