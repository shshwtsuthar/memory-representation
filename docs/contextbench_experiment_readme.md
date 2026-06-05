# ContextBench Prior-Trajectory Memory Experiment

## Purpose

This project studies whether a coding agent solves a current software-engineering issue better when it is given prior experience from an officially related software-engineering issue.

The core research question is:

```text
Does the representation of prior SWE-agent experience affect downstream coding-agent performance?
```

The experiment compares four prompt conditions. The target issue is the same across the four conditions. The only intended experimental difference is the representation of the prior issue trajectory in the `PRIOR_CONTEXT` section.

The high-level hypothesis was that a shorter deterministic memory might outperform long raw or ADP trajectory context because it should preserve the relevant prior evidence while reducing distraction and long-context retrieval burden. The final result did not cleanly support that stronger hypothesis: all prior-context conditions beat no-memory, but raw trajectory context performed best in this model/OpenHands setup.

## Dataset And Task Structure

The experiment uses official SWEContextBench / ContextBench prior-target relationships.

Each example has:

```text
prior issue A + prior SWE-agent trajectory
  -> officially related target issue B
```

The prior-target mapping is not manually guessed. It comes from the SWEContextBench relationship data.

For each target issue, the pipeline renders four OpenHands prompts:

```text
target B / no_memory
target B / raw
target B / adp
target B / memory
```

The final prompt set contained:

```text
96 targets x 4 conditions = 384 prompts
```

One target, `django__django-28147`, was later excluded from final evaluation because its sandbox image did not contain the expected manifest base commit. The final corrected evaluation therefore uses:

```text
95 targets x 4 conditions = 380 evaluated runs
```

## Experimental Conditions

### `no_memory`

The agent receives only the current target issue.

The prompt still contains a `PRIOR_CONTEXT` section for template parity, but that section explicitly states:

```text
No prior context is provided for this run.
```

This is the baseline. It measures OpenHands + Qwen performance on the target issue without any prior related issue evidence.

### `raw`

The agent receives a stripped raw Claude Code trajectory transcript from the officially related prior issue.

This representation preserves procedural trace evidence from the prior run, including:

- prior issue metadata and problem statement;
- selected user/assistant/tool events;
- file reads, grep/glob/search actions, and bash commands;
- selected tool results and observations;
- capped diagnostic and test output;
- ordering and local context from the prior debugging process.

The stripping step removes or suppresses harness/evaluator contamination, including:

- prediction artifacts;
- model patch fields;
- final patch diffs;
- `FAIL_TO_PASS` / `PASS_TO_PASS` oracle fields;
- Claude system reminders;
- reasoning content by default;
- `.claude` / `.token_usage` / session artifacts;
- obvious environment setup junk.

Raw is the least abstracted prior-context representation. It is longer and noisier, but it retains more procedural detail.

### `adp`

The agent receives a stripped ADP-rendered trajectory from the same officially related prior issue.

ADP is a normalized action/observation trajectory representation. It is more structured than raw JSONL and is rendered into text after deterministic stripping and capping.

The ADP condition exposes similar underlying evidence to raw, but through a normalized representation. It keeps the trajectory as a sequence of prior actions and observations rather than converting it into a compact memory.

### `memory`

The agent receives deterministic extractive memory generated from the stripped ADP trajectory.

This is not an LLM summary. It is produced deterministically by `scripts/contextbench/adp_to_memory.py`.

The memory extractor records:

- source metadata: prior trajectory id, repo, instance id, base commit, source model;
- prior problem statement;
- files inspected by the prior agent;
- files edited by the prior agent;
- search anchors: grep/glob patterns and paths;
- test, diagnostic, reproduction, build, and setup commands;
- observed test/runtime failure lines;
- relevant observation excerpts;
- source/test edit evidence;
- prior agent final message, if present;
- warnings about excluded or capped evidence.

The memory condition is much shorter than raw/ADP in initial prompt length, but it is not necessarily lower-token over the full OpenHands run because OpenHands accumulates repository exploration and tool outputs over many turns.

## Pipeline

The main scripts live under `scripts/contextbench/`.

### 1. Raw Claude JSONL To ADP

Script:

```text
scripts/contextbench/claude_code_jsonl_to_adp.py
```

This converts raw Claude Code JSONL trajectories into ADP trajectory records.

The ADP records are used as the canonical trajectory representation for later deterministic stripping and memory extraction.

### 2. Deterministic Trajectory Stripping

Script:

```text
scripts/contextbench/strip_trajectory_junk.py
```

This strips harness, environment, oracle, prediction, and submission artifacts from raw, ADP, and memory-adjacent trajectory representations.

Important outputs:

```text
data/contextbench_phase1/stripped/raw_transcripts/
data/contextbench_phase1/stripped/stripped_adp_rendered/
data/contextbench_phase1/stripped/generated_memory_text_from_stripped_adp/
```

The stripping policy is deterministic. It avoids LLM calls, embeddings, or stochastic selection.

### 3. ADP To Deterministic Memory

Script:

```text
scripts/contextbench/adp_to_memory.py
```

This converts stripped ADP trajectories into deterministic extractive memory.

The memory is designed to be injection-safe and reproducible. It extracts and renders evidence rather than paraphrasing it.

### 4. Run Manifest Construction

Script:

```text
scripts/contextbench/manifest/build_contextbench_run_manifest.py
```

This joins:

- official SWEContextBench experience issues;
- official SWEContextBench related target issues;
- official relationship mapping;
- prior trajectory metadata;
- raw / ADP / memory injection files;
- sandbox image tags;
- output directories.

The manifest contains one deterministic prior-target pair per target in the final setup.

Artifacts:

```text
data/contextbench_phase2/run_manifest.jsonl
data/contextbench_phase2/pair_manifest.jsonl
```

### 5. Prompt Rendering

Script:

```text
scripts/contextbench/render_contextbench_prompts.py
```

This renders the final OpenHands prompt for each target/condition pair.

Artifacts:

```text
data/contextbench_phase2/prompt_manifest.jsonl
data/contextbench_phase2/prompt_render_report.json
data/contextbench_phase2/forbidden_prompt_scan.txt
data/contextbench_phase2/prompts/<target_instance_id>/<condition>/prompt.txt
```

The prompt template keeps all instructions identical across conditions. Only the `PRIOR_CONTEXT` payload changes.

Prompt audit summary:

```text
prompt_count: 384
unique_targets: 96
condition_counts:
  no_memory: 96
  raw: 96
  adp: 96
  memory: 96
forbidden_hit_count: 0
too_long_count: 0
bad_condition_sets: {}
inconsistent_target_issue_hashes: {}
```

Initial prompt character lengths:

| Condition | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| `no_memory` | 3,063 | 2,388 | 1,444 | 17,178 |
| `raw` | 30,706 | 26,041 | 12,225 | 101,420 |
| `adp` | 37,566 | 31,793 | 15,675 | 124,094 |
| `memory` | 20,130 | 20,489 | 10,564 | 32,766 |

Prior-context character lengths:

| Condition | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| `no_memory` | 0 | 0 | 0 | 0 |
| `raw` | 27,606 | 22,658 | 8,699 | 99,157 |
| `adp` | 34,458 | 29,661 | 12,362 | 121,822 |
| `memory` | 17,022 | 17,106 | 6,495 | 26,243 |

### 6. OpenHands Execution

Script:

```text
scripts/run_openhands_contextbench.py
```

Target driver scripts:

```text
scripts/run_one_target.sh
scripts/run_all_targets.sh
```

The OpenHands runner:

1. reads `prompt_manifest.jsonl`;
2. selects rows by target and condition;
3. seeds a clean workspace from the SWEContextBench Docker image `/testbed`;
4. verifies workspace `HEAD` matches the manifest base commit;
5. copies the exact rendered prompt into the run directory;
6. runs OpenHands headless;
7. stages all workspace changes;
8. extracts `git diff --cached --binary --no-ext-diff`;
9. writes strict evaluator-compatible prediction JSON;
10. records run metadata and guardrail information.

Important final execution root:

```text
data/contextbench_phase2/execution_full_qwen36_65k_fix1/
```

Important generated files per run:

```text
runs/<target_instance_id>/<condition>/prompt.txt
runs/<target_instance_id>/<condition>/stdout.jsonl
runs/<target_instance_id>/<condition>/stderr.log
runs/<target_instance_id>/<condition>/run_meta.json
runs/<target_instance_id>/<condition>/patch.diff
runs/<target_instance_id>/<condition>/prediction.json
```

Condition-separated prediction files:

```text
predictions/full_no_memory_predictions.jsonl
predictions/full_raw_predictions.jsonl
predictions/full_adp_predictions.jsonl
predictions/full_memory_predictions.jsonl
predictions/contextbench_files/smoke/<condition>/<instance_id>_preds.json
```

### 7. Model And OpenHands Configuration

Model:

```text
Qwen3.6-35B-A3B via Ollama OpenAI-compatible endpoint
LLM_MODEL=openai/qwen3.6:35b-a3b-65k
```

OpenHands:

```text
openhands==1.13.1
temperature=0
top_p=1
max_iterations=2000
reasoning_effort=none
LLM timeout=1800s
```

The local Ollama model was configured with a 65K context variant. The repository also contains:

```text
scripts/contextbench/qwen_65k.Modelfile
```

with:

```text
PARAMETER num_ctx 65536
```

### 8. Runtime Fixes Applied Before Final Results

Several infrastructure issues were found during the run and fixed before the final corrected result set.

Important fixes:

- Missing `security_risk` tool field in OpenHands was defaulted to `SecurityRisk.UNKNOWN` instead of raising and burning iterations.
- OpenHands LLM timeout was raised to 1800 seconds.
- `reasoning_effort="none"` was forced for Qwen to avoid empty-content thinking/tool loops.
- Run containers are forcibly cleaned up after timeout/failure and before launching the next run.
- Workspace HEAD mismatch now fails fast instead of continuing with a wrong checkout.
- Patch collection was made binary-safe via `git diff --binary` decoding with replacement.
- Off-workspace write scanning was fixed so sibling condition directories are not treated as contamination.

The invalid target `django__django-28147` was excluded because its sandbox image did not contain the expected manifest base commit.

### 9. Evaluation

Evaluation was run with the official SWEContextBench evaluator from:

```text
/mnt/data/shashwat/SWEContextBench
```

The corrected evaluation used fresh run ids:

```text
qwen36_no_memory_95_keepimg
qwen36_raw_95_keepimg
qwen36_adp_95_keepimg
qwen36_memory_95_keepimg
```

The important evaluator flag was:

```text
--no-remove-instance-image
```

This matters because the first evaluation attempt removed Docker instance images after each condition and later conditions hit `Hardened image not found` errors. The corrected pass kept instance images and completed cleanly.

Corrected evaluator health:

```text
all four conditions completed with rc=0
Hardened image not found: 0
```

Corrected evaluation wall-clock:

| Condition | Evaluation Wall Clock |
|---|---:|
| `no_memory` | 2.13h |
| `raw` | 2.31h |
| `adp` | 2.15h |
| `memory` | 2.23h |
| Total corrected evaluation | 8.82h |

## Final Results

Final corrected evaluation results over 95 valid targets:

| Condition | Resolved | Success Rate | Non-Empty Patches | Empty Patches | Eval No Patch | Eval Patch Failed | Other Eval Error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_memory` | 10 / 95 | 10.5% | 62 | 33 | 33 | 6 | 0 |
| `raw` | 19 / 95 | 20.0% | 82 | 13 | 13 | 7 | 0 |
| `adp` | 15 / 95 | 15.8% | 81 | 14 | 14 | 10 | 1 |
| `memory` | 16 / 95 | 16.8% | 79 | 16 | 16 | 4 | 0 |

The one nonstandard ADP evaluator error was:

```text
sympy__sympy-19006:
Verifier setup failed: collected 0 tests after patch; expected 52 requested tests
```

Final ordering:

```text
raw > memory > adp > no_memory
19    16       15    10
```

## Token And Runtime Metrics

Token counts below are accumulated OpenHands LLM tokens across all turns, extracted from persisted OpenHands conversation state. They are not just the initial prompt token counts.

| Condition | Input Tokens | Output Tokens | Total Tokens | LLM Calls | OpenHands Wall Total | Mean Runtime | Median Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_memory` | 149,116,755 | 4,677,579 | 153,794,334 | 4,416 | 45.20h | 28.5m | 13.3m |
| `raw` | 154,667,541 | 3,310,983 | 157,978,524 | 3,079 | 34.86h | 22.0m | 12.4m |
| `adp` | 177,160,048 | 4,886,695 | 182,046,743 | 2,880 | 45.29h | 28.6m | 14.1m |
| `memory` | 147,791,422 | 3,843,866 | 151,635,288 | 2,780 | 41.17h | 26.0m | 11.3m |

Initial prompt size was much smaller for memory than raw/ADP, but total accumulated input tokens were similar across conditions because OpenHands repeatedly re-sends growing conversation context containing repository exploration, tool results, and agent state.

Maximum per-turn token counts reached approximately the configured long-context range in all conditions:

| Condition | Median Max Per-Turn Tokens | Max Per-Turn Tokens |
|---|---:|---:|
| `no_memory` | 39,370 | 98,304 |
| `raw` | 48,160 | 98,307 |
| `adp` | 51,035 | 98,306 |
| `memory` | 45,419 | 98,305 |

## Interpretation

The robust finding is:

```text
related prior trajectory context improves performance over no prior context
```

All three prior-context conditions beat `no_memory`:

```text
raw:    +9 resolved over no_memory
memory: +6 resolved over no_memory
adp:    +5 resolved over no_memory
```

The representation also matters. Raw performed best in this setup even though it is longer than memory.

One plausible interpretation is that raw preserves procedural evidence that the model can exploit:

- exact prior search paths;
- command sequence;
- repeated cues that reinforce relevance;
- failed hypotheses;
- concrete tool observations;
- patch-shape hints;
- local file/path wording.

Memory preserves the most explicit extracted evidence, but it may compress away useful procedural signal. ADP is structured and normalized, but that structure was not enough to outperform raw for this model/OpenHands stack.

This does not prove that raw is universally best. It shows that, for this OpenHands + Qwen3.6-35B-A3B setup, raw prior trajectory context was the strongest representation, while deterministic memory and ADP were still beneficial relative to no prior context.

## Main Caveats

- Final evaluation covers 95 targets, not 96, because `django__django-28147` had an invalid sandbox-image/base-commit mismatch.
- The first evaluator pass was contaminated by Docker image cleanup and should be ignored.
- The corrected evaluator pass used `--no-remove-instance-image` and had zero hardened-image failures.
- The absolute margins between raw, memory, and ADP are modest, so the ordering should be treated as an observed result for this model/tooling stack, not a universal claim.
- Evaluation can have small nondeterminism/flakiness at the one-instance level.
- Token totals are accumulated OpenHands conversation tokens, not initial prompt tokens.

## Bottom Line

The experiment supports the main research question:

```text
The representation of prior SWE-agent experience affects downstream coding-agent performance.
```

The result is not simply that shorter context wins. In this run, raw trajectory context was most effective, despite being longer than memory. The likely reason is that raw retained procedural debugging evidence that the model could use directly.

