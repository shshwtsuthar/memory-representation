# Memory Representation

This repository contains code and analysis artifacts for studying how different representations of prior software-engineering agent trajectories affect downstream coding-agent issue resolution, localization, and tool behavior.

## Paper

Paper title: "Memory as a Map: Prior-Trajectory Representations for Software Engineering Agents"

Paper: forthcoming / workshop submission.

## What this repository contains

- [scripts/contextbench/](scripts/contextbench/): conversion, stripping, rendering, and analysis scripts for the ContextBench/SWEContextBench prior-trajectory experiment.
- [scripts/nebius-experiment/](scripts/nebius-experiment/): OpenHands-to-ADP conversion and validation utilities for the auxiliary Nebius trajectory artifact.
- [artifacts/](artifacts/): generated or converted artifact metadata, including the Nebius OpenHands ADP manifest.
- [data/](data/): local metadata, manifests, generated analysis data, and local experiment artifacts.
- [reports/](reports/): overlap and schema audit reports.
- [figures/](figures/): figure-generation code, figure outputs, and validation notes.
- [docs/](docs/): additional documentation for artifacts, reproducibility, the pipeline, and release notes.
- [analysis/](analysis/): post hoc analysis work products when present locally.

Large raw trajectories, run outputs, and large dataset files should not be committed to Git. Use the released Hugging Face artifacts for reproducibility data.

## Core experiment

The experiment compares four prior-context conditions:

- None: no related prior context.
- Trace: stripped Claude Code transcript replay.
- Action: stripped ADP-normalized action/observation trace rendered as text.
- Digest: deterministic extractive evidence digest.

The target issue and task instructions are held fixed across conditions; only `PRIOR_CONTEXT` changes.

The original rendered prompt set contained 96 targets x 4 conditions = 384 prompts. The target `django__django-28147` was excluded because the sandbox image did not contain the expected manifest base commit. The evaluated set is therefore 95 valid targets x 4 conditions = 380 evaluated runs.

The agent setup is OpenHands 1.13.1 with Qwen3.6-35B-A3B through an Ollama OpenAI-compatible endpoint, temperature 0, top_p 1, and max_iterations 2000. ADP is used as a normalization backbone and as the basis for the Action representation; the prompt does not show raw ADP JSON.

## Main results

Trace has the highest observed fixed-condition solve count in this OpenHands/Qwen setup: 19/95. Digest solves 16/95, Action solves 15/95, and None solves 10/95.

Digest and Action solve partially different targets. A retrospective prior-context oracle reaches 28/95, showing +9 target headroom over the best fixed prior-context representation. The main representation-level result is complementarity, not universal trace superiority.

These results should be read as representation evidence in this specific setup, not as a universal ranking or causal mechanism proof.

## Data and artifacts

- [ContextBench/SWEContextBench processed artifacts](https://huggingface.co/datasets/shshwtsuthar/memory-representation-contextbench-artifacts): `shshwtsuthar/memory-representation-contextbench-artifacts`
- [ContextBench/SWEContextBench trace artifacts](https://huggingface.co/datasets/shshwtsuthar/memory-representation-contextbench-traces): `shshwtsuthar/memory-representation-contextbench-traces`
- [Nebius OpenHands ADP v0.1](https://huggingface.co/datasets/shshwtsuthar/memory-representation-nebius-openhands-adp-v0.1): `shshwtsuthar/memory-representation-nebius-openhands-adp-v0.1`

The ContextBench/SWEContextBench artifact is the primary processed reproducibility artifact. The trace artifact contains raw or reformatted trajectories for users who need inspection-level provenance. The Nebius OpenHands->ADP artifact is auxiliary converter validation and is not part of the 95-target experiment.

## Quick start

This repository does not currently include a pinned project environment. For documentation checks and figure/table utilities, create a minimal environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
python scripts/release/validate_repo_docs.py
```

To validate the release documentation through `make`:

```bash
make validate-docs
```

## Reproducing tables and figures

Generated figures can be rebuilt from the local processed CSVs:

```bash
python figures/make_figures.py --data-dir data --out-dir figures
```

The release documentation and artifact URL checks can be run with:

```bash
python scripts/release/validate_repo_docs.py
python scripts/release/check_artifact_links.py
python scripts/release/summarize_release_artifacts.py
```

Full re-execution requires the SWEContextBench/OpenHands environment and local run artifacts. The released processed artifacts are intended for reproducing tables and figures without rerunning all agents.

## Citation

See [CITATION.cff](CITATION.cff). The paper citation is preliminary until the workshop or archival paper metadata is final.

## License

License: TODO. Code and data may have different licenses. The Hugging Face dataset cards document artifact-specific licensing and upstream attribution.

See [docs/LICENSE_DECISION.md](docs/LICENSE_DECISION.md) for the manual license decision that remains before a public release.

## Limitations

- Single OpenHands/Qwen setup.
- No unrelated-prior or length-matched distractor-prior control.
- Mechanism labels are post hoc.
- Retrospective oracle is diagnostic, not deployable.
