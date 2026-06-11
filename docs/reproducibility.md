# Reproducibility

This project has three reproducibility levels.

## From processed artifacts

The processed ContextBench/SWEContextBench artifact is intended for reproducing tables, figures, prompt manifests, and condition-level analyses without rerunning all agents:

https://huggingface.co/datasets/shshwtsuthar/memory-representation-contextbench-artifacts

With the processed CSVs available locally, figures can be rebuilt with:

```bash
python figures/make_figures.py --data-dir data --out-dir figures
```

The documentation validation checks can be run with:

```bash
python scripts/release/validate_repo_docs.py
```

## From trace artifacts

The trace artifact supports inspection of raw or reformatted trajectories:

https://huggingface.co/datasets/shshwtsuthar/memory-representation-contextbench-traces

Use it when checking prompt provenance, stripped transcript behavior, or representation differences. It is not needed for every table or figure.

## By rerunning OpenHands

Full re-execution requires the SWEContextBench/OpenHands environment and local run artifacts. It also requires the benchmark/docker setup, matching sandbox images, the OpenHands 1.13.1 runtime, and a Qwen3.6-35B-A3B endpoint configured through an Ollama OpenAI-compatible API.

The original rendered prompt set contained 96 targets x 4 conditions = 384 prompts. The final evaluated set excludes `django__django-28147` because the sandbox image did not contain the expected manifest base commit, leaving 95 valid targets x 4 conditions = 380 evaluated runs.

Reruns may differ if benchmark images, dependencies, model serving, OpenHands behavior, or external package indexes change.

## Auxiliary converter validation

The Nebius OpenHands->ADP artifact validates the converter on a large external OpenHands trajectory set:

https://huggingface.co/datasets/shshwtsuthar/memory-representation-nebius-openhands-adp-v0.1

It is auxiliary converter validation and is not part of the 95-target ContextBench/SWEContextBench experiment.
