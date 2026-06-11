# Overlap Results

## Official Gold Patch Availability
No official `patch`, `gold_patch`, or `test_patch` fields were found in the local phase-2/dataset manifests scanned. Generated model patches were not used as gold patches.

## Gold Overlap Buckets
| bucket | count |
|---|---|
| unknown_gold_overlap | 95 |

## Prior Trajectory Localization Buckets
| bucket | count |
|---|---|
| unknown | 95 |

## Bucket Summary
| bucket_type | bucket | condition | n_targets | resolved_count | success_rate | non_empty_patch_count | patch_attempt_rate | mean_runtime | mean_total_tokens |
|---|---|---|---|---|---|---|---|---|---|
| overlap_bucket | unknown_gold_overlap | no_memory | 95 |  |  |  |  |  |  |
| overlap_bucket | unknown_gold_overlap | raw | 95 |  |  |  |  |  |  |
| overlap_bucket | unknown_gold_overlap | adp | 95 |  |  |  |  |  |  |
| overlap_bucket | unknown_gold_overlap | memory | 95 |  |  |  |  |  |  |
| localization_bucket | unknown | no_memory | 95 |  |  |  |  |  |  |
| localization_bucket | unknown | raw | 95 |  |  |  |  |  |  |
| localization_bucket | unknown | adp | 95 |  |  |  |  |  |  |
| localization_bucket | unknown | memory | 95 |  |  |  |  |  |  |

Interpretation should remain descriptive until official target/prior gold patches are available. Current local output can support prior trajectory file-activity summaries but cannot separate broad transfer from path leakage/localization with gold-file evidence.
