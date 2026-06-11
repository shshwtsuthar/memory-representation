# Run Discovery

Timestamp: `20260606_220027`

Remote discovery used `sshpass -e ssh` with `SSHPASS` supplied by the environment. No password was written to scripts, reports, or logs.

## Search Roots

- `/mnt/data/shashwat/memory-representation`
- `/mnt/data/shashwat/openhands-adp-memory`

No search was run outside these roots.

## Final Execution Root Found

`/mnt/data/shashwat/openhands-adp-memory/data/contextbench_phase2/execution_full_qwen36_65k_fix1`

## Discovered Final Run Artifacts

From `remote/discovery.tsv`:

| artifact | count |
|---|---:|
| `run_meta.json` | 381 |
| `prediction.json` | 380 |
| `patch.diff` | 380 |
| `stdout.jsonl` | 380 |
| `stderr.log` | 380 |
| OpenHands `base_state.json` | 380 |

The extra `run_meta.json` corresponds to the excluded `django__django-28147/no_memory` base-commit mismatch run.

## Full Prediction JSONL Files Found

- `/mnt/data/shashwat/openhands-adp-memory/data/contextbench_phase2/execution_full_qwen36_65k_fix1/predictions/full_no_memory_predictions.jsonl`
- `/mnt/data/shashwat/openhands-adp-memory/data/contextbench_phase2/execution_full_qwen36_65k_fix1/predictions/full_raw_predictions.jsonl`
- `/mnt/data/shashwat/openhands-adp-memory/data/contextbench_phase2/execution_full_qwen36_65k_fix1/predictions/full_adp_predictions.jsonl`
- `/mnt/data/shashwat/openhands-adp-memory/data/contextbench_phase2/execution_full_qwen36_65k_fix1/predictions/full_memory_predictions.jsonl`

Each has 95 lines.

## Evaluator Search Result In Original Roots

The requested final evaluator run ids were searched under the two allowed roots:

- `qwen36_no_memory_95_keepimg`
- `qwen36_raw_95_keepimg`
- `qwen36_adp_95_keepimg`
- `qwen36_memory_95_keepimg`

No matching final evaluator output path was discovered in the two originally allowed roots.

Targeted searches for evaluator-like files found only:

- earlier astropy smoke/envcheck reports under `/mnt/data/shashwat/memory-representation/logs/run_evaluation`
- `/mnt/data/shashwat/memory-representation/data/contextbench_phase2/evaluation/smoke_astropy__astropy-15082_dataset.jsonl`
- `/mnt/data/shashwat/memory-representation/data/contextbench_phase2/evaluation/smoke_astropy__astropy-15082_dataset_harness.jsonl`

Content searches for `resolved`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `Verifier setup failed`, and the qwen run ids did not find final 95-target evaluator result files in the allowed roots.

## Authorized Evaluator Root

The user then explicitly authorized read-only access to:

`/mnt/data/shashwat/SWEContextBench`

Final evaluator summaries were found:

- `/mnt/data/shashwat/SWEContextBench/qwen36_no_memory_95_keepimg.json`
- `/mnt/data/shashwat/SWEContextBench/qwen36_raw_95_keepimg.json`
- `/mnt/data/shashwat/SWEContextBench/qwen36_adp_95_keepimg.json`
- `/mnt/data/shashwat/SWEContextBench/qwen36_memory_95_keepimg.json`

Per-target evaluator reports were also found under:

`/mnt/data/shashwat/SWEContextBench/logs/run_evaluation/<run_id>/openhands-qwen3.6-35b-a3b-65k-ollama-contextbench-memory-repr/<instance_id>/report.json`

These evaluator summaries and reports were used as the source of truth for `resolved_*`, patch application status, `No patch`, patch-failed, and verifier error fields.

## Local Manifests And Gold Patch Data

Local files found and inspected:

- `data/contextbench_phase2/run_manifest.jsonl`
- `data/contextbench_phase2/pair_manifest.jsonl`
- `data/contextbench_phase2/prompt_manifest.jsonl`
- `data/contextbench_phase2/prompt_render_report.json`
- `data/contextbench_phase2/forbidden_prompt_scan.txt`
- `data/contextbench_dataset/data/SWEContextBench_Related_Lite.parquet`
- `data/contextbench_dataset/data/SWEContextBench_Lite_Experience.parquet`
- `data/contextbench_dataset/data/SWEContextBench_Relationship.parquet`

The parquet dataset includes official `patch`, `test_patch`, `FAIL_TO_PASS`, and `PASS_TO_PASS` fields, so official gold-patch overlap is not blocked. The blocking missing data is the final evaluator target-level resolved labels.

## Raw Discovery Files

- `remote/discovery.tsv`
- `remote/evaluator_candidate_files.tsv`
- `remote/evaluator_candidate_dirs.txt`
- `remote/evaluator_content_hits.txt`
- `remote/resolved_content_hits.txt`
- `remote/evaluation_tree.tsv`
- `remote/sample_run_schema.txt`
- `remote/final_report_prediction_schema.txt`
