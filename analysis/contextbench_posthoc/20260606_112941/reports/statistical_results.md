# Statistical Results

Paired statistical tests were not run because `paired_results.csv` lacks complete evaluator truth (380 missing condition-target outcomes).

## Patch Attempt Analysis
| condition | n | empty_patch_count | non_empty_patch_count | patch_attempt_rate | resolved_count | resolved_given_attempt_rate | unresolved_attempt_count | eval_patch_failed_count | eval_no_patch_count | other_eval_error_count |
|---|---|---|---|---|---|---|---|---|---|---|
| no_memory | 95 |  |  |  |  |  |  |  |  |  |
| raw | 95 |  |  |  |  |  |  |  |  |  |
| adp | 95 |  |  |  |  |  |  |  |  |  |
| memory | 95 |  |  |  |  |  |  |  |  |  |

## Patch Attempt Pairwise McNemar
_No rows available._

## Runtime / Token Accounting
| condition | verification_status | observed_total_tokens | expected_total_tokens | observed_total_llm_calls | expected_total_llm_calls | observed_total_wall_hours | expected_total_wall_hours | expected_max_turn_tokens |
|---|---|---|---|---|---|---|---|---|
| no_memory | unverified_missing_artifacts |  | 153794334 |  | 4416 |  | 45.2 | 98304 |
| raw | unverified_missing_artifacts |  | 157978524 |  | 3079 |  | 34.86 | 98307 |
| adp | unverified_missing_artifacts |  | 182046743 |  | 2880 |  | 45.29 | 98306 |
| memory | unverified_missing_artifacts |  | 151635288 |  | 2780 |  | 41.17 | 98305 |

The expected max per-turn token values are around 98K despite the 65K context configuration label. This run did not have the full OpenHands accounting artifacts needed to determine whether that reflects tokenizer mismatch, prompt+completion accounting, rolling context, or actual served context.
