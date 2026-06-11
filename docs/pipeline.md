# Pipeline

This page documents the release-facing pipeline. Only paths that exist in this repository are listed.

## 1. Claude Code JSONL -> canonical ADP

Script: [scripts/contextbench/claude_code_jsonl_to_adp.py](../scripts/contextbench/claude_code_jsonl_to_adp.py)

Raw Claude Code JSONL trajectories are converted into canonical ADP records. ADP is the normalization backbone for later stripping, rendering, and deterministic memory extraction.

## 2. ADP stripping

Script: [scripts/contextbench/strip_trajectory_junk.py](../scripts/contextbench/strip_trajectory_junk.py)

The stripping step removes harness, evaluator, oracle, prediction, submission, environment, and other release-inappropriate trajectory material. The same deterministic policy supports the Trace, Action, and Digest prompt conditions.

## 3. Trace rendering

Script: [scripts/contextbench/strip_trajectory_junk.py](../scripts/contextbench/strip_trajectory_junk.py)

The Trace condition renders a stripped Claude Code transcript replay as text. It preserves procedural evidence from the prior issue while suppressing evaluator and benchmark leakage.

## 4. Action rendering

Script: [scripts/contextbench/strip_trajectory_junk.py](../scripts/contextbench/strip_trajectory_junk.py)

The Action condition renders stripped ADP-normalized action/observation traces as text. The prompt exposes the normalized trajectory evidence, not raw ADP JSON.

## 5. Digest rendering

Script: [scripts/contextbench/adp_to_memory.py](../scripts/contextbench/adp_to_memory.py)

The Digest condition is a deterministic extractive evidence digest built from stripped ADP. It extracts prior problem text, inspected and edited files, search anchors, commands, failures, observations, and edit evidence without an LLM summarization step.

## 6. Prompt rendering and OpenHands target execution

Prompt rendering script: [scripts/contextbench/render_contextbench_prompts.py](../scripts/contextbench/render_contextbench_prompts.py)

Run scripts:

- [scripts/run_openhands_contextbench.py](../scripts/run_openhands_contextbench.py)
- [scripts/run_one_target.sh](../scripts/run_one_target.sh)
- [scripts/run_all_targets.sh](../scripts/run_all_targets.sh)
- [scripts/launch_contextbench_tmux.sh](../scripts/launch_contextbench_tmux.sh)

The rendered prompt holds the target issue and task instructions fixed across conditions. Only `PRIOR_CONTEXT` changes.

The evaluated agent setup was OpenHands 1.13.1 with Qwen3.6-35B-A3B through an Ollama OpenAI-compatible endpoint, temperature 0, top_p 1, and max_iterations 2000.

## 7. Evaluator/results analysis

Local processed CSVs in [data/](../data/) and figure-generation code in [figures/](../figures/) support the release figures and summary tables. The available figure command is:

```bash
python figures/make_figures.py --data-dir data --out-dir figures
```

Full re-execution depends on the SWEContextBench/OpenHands environment, benchmark images, model endpoint, and local run artifacts.
