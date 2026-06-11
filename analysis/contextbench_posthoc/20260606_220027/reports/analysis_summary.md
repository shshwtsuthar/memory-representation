# Analysis Summary

## Data Found

- Final execution root: `/mnt/data/shashwat/openhands-adp-memory/data/contextbench_phase2/execution_full_qwen36_65k_fix1`
- Authorized evaluator root: `/mnt/data/shashwat/SWEContextBench`
- Final evaluator summaries found for all four `qwen36_*_95_keepimg` conditions.
- Final run artifacts found for 380 evaluated runs.
- Official gold patches found in local SWEContextBench parquet files.

## Main Aggregate Results

|condition|n_targets|resolved_count|success_rate|non_empty_patch_count|empty_patch_count|
|---|---|---|---|---|---|
|no_memory|95|10|0.1053|62|33|
|raw|95|19|0.2|82|13|
|adp|95|15|0.1579|81|14|
|memory|95|16|0.1684|79|16|

## Paired Statistical Results

|condition_a|condition_b|n|a_resolved_count|b_resolved_count|rate_diff_a_minus_b|a_only_resolved|b_only_resolved|mcnemar_exact_p|holm_adjusted_p_across_six_comparisons|
|---|---|---|---|---|---|---|---|---|---|
|raw|no_memory|95|19|10|0.09474|11|2|0.02246|0.1348|
|memory|no_memory|95|16|10|0.06316|9|3|0.146|0.73|
|adp|no_memory|95|15|10|0.05263|8|3|0.2266|0.9062|
|raw|memory|95|19|16|0.03158|9|6|0.6072|1|
|raw|adp|95|19|15|0.04211|10|6|0.4545|1|
|memory|adp|95|16|15|0.01053|7|6|1|1|

## Patch Attempt Results

|condition|n|empty_patch_count|non_empty_patch_count|patch_attempt_rate|resolved_count|resolved_given_attempt_rate|eval_patch_failed_count|eval_no_patch_count|
|---|---|---|---|---|---|---|---|---|
|no_memory|95|33|62|0.6526|10|0.1613|6|33|
|raw|95|13|82|0.8632|19|0.2317|7|13|
|adp|95|14|81|0.8526|15|0.1852|10|14|
|memory|95|16|79|0.8316|16|0.2025|4|16|

## Sensitivity

`sympy__sympy-19006` ADP verifier setup failure is counted unresolved in the main analysis. Sensitivity excluding the target from all conditions is written to `data/pairwise_mcnemar_exclude_sympy19006.csv`.

## Prompt Audit

- prompt count: 384
- condition counts: {'adp': 96, 'memory': 96, 'no_memory': 96, 'raw': 96}
- forbidden hit count: 0
- inconsistent target issue hashes: {}

## Token And Runtime Accounting

Input/output/total token totals, wall-clock totals, mean/median runtimes, and max per-turn token values match the documented expected values from persisted OpenHands state. `condition_summary.csv` reports `llm_calls` from `run_meta.json`; this equals the documented no-memory total but is higher for raw/ADP/memory than the documented LLM-call totals. The discrepancy is retained as a diagnostic rather than overwritten because the artifacts expose multiple plausible call-count definitions (`run_meta.llm_call_count`, action counts, and persisted state token usage entries).

## Recommended Paper Framing

Officially related prior SWE-agent experience improves downstream OpenHands/Qwen3.6 performance over no prior context. Raw trajectory context achieved the highest observed solve count, but differences among raw, ADP, and deterministic memory require paired statistical interpretation and should not be treated as universally established. The mechanism appears to involve localization/procedural transfer to the extent supported by overlap and transcript evidence.

## Claude-Requested Analyses

Now populated: paired target-level analysis, McNemar/bootstrap comparisons, solve-pattern analysis, patch-attempt analysis, official gold-patch overlap/localization analysis, transcript behavior mining, qualitative case selection, and sensitivity excluding `sympy__sympy-19006`.