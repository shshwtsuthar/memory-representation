# Generated Figures

Run:

```bash
python figures/make_figures.py --data-dir data --out-dir figures
```

The script validates the expected 95-target analysis counts before plotting:
condition solve counts, non-empty patch counts, solve-pattern counts, oracle
counts, exact McNemar/bootstrap columns, and the 95 x 4 time-to-gold-file
structure. It fails loudly if any required CSV is missing or any headline count
drifts.

The plots use pandas and matplotlib only. PDFs are vector outputs intended for
LaTeX inclusion; PNGs are high-resolution previews. The palette is colorblind
safe and avoids decorative effects.

## Outputs

- fig_paired_effects.pdf / fig_paired_effects.png: Resolution gains versus None are descriptive and paired-success tests are not Holm-significant; non-empty patch-attempt effects are large and Holm-significant.
- fig_solve_complementarity_oracle.pdf / fig_solve_complementarity_oracle.png: Trace has the highest fixed prior-context solve count, but different representations solve different targets; the +9 prior-context oracle headroom is retrospective.
- fig_localization_map.pdf / fig_localization_map.png: Prior-context gains concentrate when prior and target are local, and prior-context runs reach target-gold files earlier with fewer mean search commands, consistent with localization/procedural transfer.
- fig_patch_action_funnel.pdf / fig_patch_action_funnel.png: Prior-context conditions mainly reduce empty-patch behavior and increase target-relevant actions; final resolution is a narrower endpoint.
- fig_disagreement_timeline_raster.pdf / fig_disagreement_timeline_raster.png: Different memory representations route the same target through different evidence and action paths, explaining why aggregate solve counts hide complementarity.
- appendix_fig_evidence_action_heatmap.pdf / appendix_fig_evidence_action_heatmap.png: Representation-exposed evidence is common, but evidence presence alone is not sufficient for success; the agent must act on it.

## Scientific Notes

- This is not a leaderboard figure set.
- The retrospective oracle is labeled as retrospective and not deployable.
- Mechanism figures use the wording "consistent with localization/procedural transfer" and do not claim causality.
- The action funnel is not a causal pipeline because test-running is not strictly sequential after editing.
- The timeline raster uses structured event data and omits transcript snippets from the main figure.

## Remote Transcript Provenance

Selected run directories under `/mnt/data/shashwat/openhands-adp-memory/data/contextbench_phase2/execution_full_qwen36_65k_fix1/runs` were checked read-only for the disagreement cases. The small local provenance file `remote_artifact_check.tsv`, when present, records artifact filenames and sizes. Raw transcript text is not embedded in the figures.

## Validation Summary

```json
{
  "adp_verifier_anomaly_counted_unresolved": "sympy__sympy-19006",
  "best_fixed_prior": 19,
  "condition_counts": {
    "adp": {
      "n_targets": 95,
      "non_empty_patch_count": 81,
      "resolved_count": 15
    },
    "memory": {
      "n_targets": 95,
      "non_empty_patch_count": 79,
      "resolved_count": 16
    },
    "no_memory": {
      "n_targets": 95,
      "non_empty_patch_count": 62,
      "resolved_count": 10
    },
    "raw": {
      "n_targets": 95,
      "non_empty_patch_count": 82,
      "resolved_count": 19
    }
  },
  "excluded_target": "django__django-28147",
  "localization_bucket_counts": {
    "prior_trajectory_edited_target_gold_file": 38,
    "prior_trajectory_inspected_target_gold_file": 33,
    "prior_trajectory_never_touched_target_gold_area": 11,
    "prior_trajectory_same_directory_as_target_gold": 13
  },
  "oracle_all": 29,
  "oracle_headroom_over_best_fixed_prior": 9,
  "oracle_prior": 28,
  "overlap_bucket_counts": {
    "no_gold_file_overlap": 17,
    "same_directory_only": 15,
    "same_file_overlap": 63
  },
  "patch_attempt_counts": {
    "adp": {
      "empty_patch_count": 14,
      "non_empty_patch_count": 81
    },
    "memory": {
      "empty_patch_count": 16,
      "non_empty_patch_count": 79
    },
    "no_memory": {
      "empty_patch_count": 33,
      "non_empty_patch_count": 62
    },
    "raw": {
      "empty_patch_count": 13,
      "non_empty_patch_count": 82
    }
  },
  "solve_pattern_counts": {
    "adp_only": 2,
    "all_failed": 66,
    "all_four_solved": 5,
    "memory_only": 3,
    "no_memory_only_all_prior_fail": 1,
    "raw_only": 6
  },
  "time_to_gold_rows": 380,
  "timeline_event_rows": 28583
}
```
