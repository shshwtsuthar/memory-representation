# Artifacts

The repository is intentionally small relative to the full experiment. Large processed artifacts and trajectories are distributed through Hugging Face dataset repositories rather than committed to Git.

## ContextBench/SWEContextBench processed artifacts

- Repo ID: `shshwtsuthar/memory-representation-contextbench-artifacts`
- URL: https://huggingface.co/datasets/shshwtsuthar/memory-representation-contextbench-artifacts

This is the primary processed reproducibility artifact for the 95-target ContextBench/SWEContextBench experiment. Use it for reproducing tables, figures, prompt manifests, and condition-level analyses without rerunning all agents.

## ContextBench/SWEContextBench trace artifacts

- Repo ID: `shshwtsuthar/memory-representation-contextbench-traces`
- URL: https://huggingface.co/datasets/shshwtsuthar/memory-representation-contextbench-traces

This artifact contains raw or reformatted trajectories for users who need inspection-level provenance. It is larger and more detailed than the processed analysis artifact.

## Nebius OpenHands ADP v0.1

- Repo ID: `shshwtsuthar/memory-representation-nebius-openhands-adp-v0.1`
- URL: https://huggingface.co/datasets/shshwtsuthar/memory-representation-nebius-openhands-adp-v0.1

This artifact supports auxiliary OpenHands->ADP converter validation. It is not part of the 95-target ContextBench/SWEContextBench experiment. The local manifest is [artifacts/nebius_openhands_adp_v0.1/MANIFEST.md](../artifacts/nebius_openhands_adp_v0.1/MANIFEST.md).
