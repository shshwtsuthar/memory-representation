# ContextBench Posthoc Analysis Summary

## Data Found
- Prompt manifest rows: 384
- Unique prompt targets: 96
- Prompt condition counts: `{'adp': 96, 'memory': 96, 'no_memory': 96, 'raw': 96}`
- Forbidden prompt hit count: `0`
- Inconsistent target issue hashes across conditions: `{}`

## Data Missing / Blocked
- Evaluator result files for qwen36_*_95_keepimg were not found locally; remote SSH authentication is required to build resolved_* fields.
- Full execution artifact root was not found/provided; run metadata, token accounting, patches, and transcripts are mostly unavailable.
- Remote paths under ['/mnt/data/shashwat/memory-representation', '/mnt/data/shashwat/openhands-adp-memory'] were not discovered by the script in this run.

## Main Aggregate Results
Evaluator-backed aggregate counts from the paired table:
- `no_memory`: unavailable; 95 missing evaluator values
- `raw`: unavailable; 95 missing evaluator values
- `adp`: unavailable; 95 missing evaluator values
- `memory`: unavailable; 95 missing evaluator values

Expected aggregate counts supplied in the request:
- `no_memory`: 10 / 95
- `raw`: 19 / 95
- `adp`: 15 / 95
- `memory`: 16 / 95

## Validation
- no_memory resolved count cannot be validated: 95/95 rows lack evaluator truth.
- raw resolved count cannot be validated: 95/95 rows lack evaluator truth.
- adp resolved count cannot be validated: 95/95 rows lack evaluator truth.
- memory resolved count cannot be validated: 95/95 rows lack evaluator truth.

## Paired Statistical Results
Exact McNemar tests, paired bootstrap CIs, solve-pattern counts, and target flip tables were not inferentially populated because evaluator-backed target-condition outcomes are missing locally. The CSV files exist with the requested schemas; see `data/pairwise_mcnemar.csv`, `data/paired_bootstrap_cis.csv`, `data/solve_patterns.csv`, and `data/target_flip_table.csv`.

## Patch-Attempt Results
Patch-attempt rates and conditional patch quality require final `patch.diff`, `prediction.json`, evaluator no-patch/patch-failed statuses, or equivalent full run artifacts. Those were not available locally, so `data/patch_attempt_summary.csv`, `data/patch_attempt_pairwise_mcnemar.csv`, and `data/patch_quality_conditional.csv` are schema-correct but not populated with final-run values.

## Overlap / Leakage / Localization Results
The local deterministic memory file contains structured prior-trajectory file activity, so `data/overlap_features.csv` includes prior inspected/edited/test/source file sets. Official prior and target gold patch fields were not found in local manifests or dataset-style files, and generated model patches were not used as gold patches. Therefore all gold-overlap buckets are currently `unknown_gold_overlap`; this analysis cannot yet distinguish broad transfer from localization/path leakage.

## Transcript Behavior Results
Full OpenHands stdout/stderr/conversation artifacts for the 95 evaluated targets were not available locally. `data/transcript_behavior_features.csv`, command/file summaries, and failure-signature tables are present with requested schemas, but behavior counts are not populated from the final runs.

## Qualitative Mechanisms
Qualitative case selection depends on paired outcomes and transcripts. Because both are unavailable locally, `data/qualitative_case_index.csv` is empty and `reports/qualitative_case_notes.md` documents that case selection is blocked. Mechanism labels should remain hypotheses once transcripts are available.

## Sensitivity Excluding sympy__sympy-19006
The sensitivity files `data/pairwise_mcnemar_exclude_sympy19006.csv` and `data/paired_bootstrap_cis_exclude_sympy19006.csv` exist but are empty because the main paired outcome table lacks evaluator truth.

## Token / Runtime Accounting
`data/runtime_token_summary.csv` records the expected aggregate token/runtime totals supplied in the request and marks them `unverified_missing_artifacts`. The observed max per-turn token values around 98K exceed the nominal 65K context configuration label, but the available local artifacts do not support a causal explanation. Possible explanations remain tokenizer mismatch, prompt+completion accounting, rolling context accounting, or actual served context.

## Prompt / Context Audit
- Prompt count: 384
- Valid target count expected after exclusion: 95
- Excluded target: `django__django-28147`
- Prompt character lengths by condition: `{'adp': {'count': 96, 'min': 15675, 'median': 31793.0, 'mean': 37566.48, 'max': 124094}, 'memory': {'count': 96, 'min': 10564, 'median': 20488.5, 'mean': 20130.24, 'max': 32766}, 'no_memory': {'count': 96, 'min': 1444, 'median': 2387.5, 'mean': 3063.47, 'max': 17178}, 'raw': {'count': 96, 'min': 12225, 'median': 26041.0, 'mean': 30705.56, 'max': 101420}}`
- Prior-context character lengths by condition: `{'adp': {'count': 96, 'min': 12362, 'median': 29661.0, 'mean': 34458.01, 'max': 121822}, 'memory': {'count': 96, 'min': 6495, 'median': 17106.0, 'mean': 17021.77, 'max': 26243}, 'no_memory': {'count': 96, 'min': 0, 'median': 0.0, 'mean': 0.0, 'max': 0}, 'raw': {'count': 96, 'min': 8699, 'median': 22657.5, 'mean': 27606.09, 'max': 99157}}`
- Forbidden scan: `data/contextbench_phase2/forbidden_prompt_scan.txt` with hit count `0`
- Target issue text hashes identical across conditions: `True`
- Evidence that only PRIOR_CONTEXT differs: `True` from prompt report/hash audit.

## Recommended Paper Framing
Officially related prior SWE-agent experience improves downstream OpenHands/Qwen3.6 performance over no prior context. Raw trajectory context achieved the highest observed solve count, but differences among raw, ADP, and deterministic memory require paired statistical interpretation and should not be treated as universally established. The mechanism appears to involve localization/procedural transfer to the extent supported by overlap and transcript evidence.

## Status
Paired statistical, overlap, transcript, and qualitative sections are populated by the companion scripts when evaluator/run artifacts are present.
