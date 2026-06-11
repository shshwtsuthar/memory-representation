#!/usr/bin/env python3
"""Print expected release artifact repositories and local file names."""

from __future__ import annotations


ARTIFACTS = [
    {
        "name": "ContextBench/SWEContextBench processed artifacts",
        "repo_id": "shshwtsuthar/memory-representation-contextbench-artifacts",
        "local_files": [
            "data/contextbench_phase2/pair_manifest.jsonl",
            "data/contextbench_phase2/prompt_manifest.jsonl",
            "data/contextbench_phase2/prompt_render_report.json",
            "data/condition_summary.csv",
            "data/paired_results.csv",
            "figures/figure_manifest.json",
        ],
    },
    {
        "name": "ContextBench/SWEContextBench trace artifacts",
        "repo_id": "shshwtsuthar/memory-representation-contextbench-traces",
        "local_files": [
            "data/contextbench_phase1/all_trajectories.adp.jsonl",
            "data/contextbench_phase1/raw_cleaning_manifest.jsonl",
            "data/contextbench_phase2/run_manifest.jsonl",
        ],
    },
    {
        "name": "Nebius OpenHands ADP v0.1",
        "repo_id": "shshwtsuthar/memory-representation-nebius-openhands-adp-v0.1",
        "local_files": [
            "artifacts/nebius_openhands_adp_v0.1/MANIFEST.md",
            "data/nebius/successful_small_trajectories.csv",
        ],
    },
]


def main() -> int:
    for artifact in ARTIFACTS:
        print(artifact["name"])
        print(f"  repo_id: {artifact['repo_id']}")
        print("  expected local file names:")
        for local_file in artifact["local_files"]:
            print(f"    - {local_file}")
        print()
    print("This summary does not assume the Hugging Face datasets are downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
