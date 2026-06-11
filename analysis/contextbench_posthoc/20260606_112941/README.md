# ContextBench Posthoc Analysis Workspace

Generated analysis workspace for the ContextBench prior-trajectory memory experiment.

## Contents
- `run_discovery.md`: local/remote path discovery notes.
- `missing_data.md`: missing or blocked artifact sources.
- `scripts/`: rerunnable collection/statistics/overlap/transcript scripts.
- `data/`: CSV outputs.
- `reports/`: Markdown summaries.
- `figures/`: PNG figures when plotting inputs are available.

## Rerun
Run from the repository root:

```bash
python /home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_112941/scripts/collect_contextbench_posthoc.py --out-dir /home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_112941
python /home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_112941/scripts/stats_contextbench_posthoc.py --out-dir /home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_112941
python /home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_112941/scripts/overlap_contextbench_posthoc.py --out-dir /home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_112941
python /home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_112941/scripts/transcript_mining_contextbench_posthoc.py --out-dir /home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_112941
```

For remote discovery, set `SSHPASS` in the shell and pass `--remote`; do not place the password in scripts or reports.
