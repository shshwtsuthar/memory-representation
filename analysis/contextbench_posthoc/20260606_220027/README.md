# ContextBench Posthoc Extraction

This run is the populated posthoc analysis for the ContextBench prior-trajectory memory experiment.

Remote discovery first searched only the two originally allowed roots. After evaluator artifacts were missing there, the user explicitly authorized read-only access to:

- `/mnt/data/shashwat/SWEContextBench`

The final evaluator summaries were then found and used as the source of truth for target-level `resolved_*` labels.

Key outputs:

- `data/paired_results.csv`
- `data/condition_summary.csv`
- `data/pairwise_mcnemar.csv`
- `data/solve_patterns.csv`
- `data/patch_attempt_summary.csv`
- `data/overlap_features.csv`
- `data/transcript_behavior_features.csv`
- `reports/analysis_summary.md`
- `reports/paper_ready_tables.md`

Validation status:

- 95 paired targets.
- `django__django-28147` excluded.
- All four conditions present for each target.
- Resolved counts match `10 / 19 / 15 / 16`.
- Transcript features populated for 380 final runs.

