# Missing Data

This file records fields that could not be derived from available artifacts.

- Evaluator result files for qwen36_*_95_keepimg were not found locally; remote SSH authentication is required to build resolved_* fields.
- Full execution artifact root was not found/provided; run metadata, token accounting, patches, and transcripts are mostly unavailable.
- Remote paths under ['/mnt/data/shashwat/memory-representation', '/mnt/data/shashwat/openhands-adp-memory'] were not discovered by the script in this run.

## Validation Notes
- no_memory resolved count cannot be validated: 95/95 rows lack evaluator truth.
- raw resolved count cannot be validated: 95/95 rows lack evaluator truth.
- adp resolved count cannot be validated: 95/95 rows lack evaluator truth.
- memory resolved count cannot be validated: 95/95 rows lack evaluator truth.
