# Oracle Mechanism Report

## Validation

- Valid targets: 95
- Excluded target: django__django-28147
- no_memory: 10 / 95
- raw: 19 / 95
- adp: 15 / 95
- memory: 16 / 95
- oracle_prior: 28 / 95 (0.295)
- oracle_all: 29 / 95 (0.305)

The prior-context oracle is retrospective and not deployable. The mechanism labels below mean "consistent with" the transcript evidence, not causal proof.

## Representation-Choice Headroom

- raw-only prior solves: 6
- memory-only prior solves: 3
- ADP-only prior solves: 3
- prior oracle solves beyond best fixed prior representation: 9
- oracle_all gain over prior-only oracle: 1

## Mechanism Counts

Among prior-oracle solves:
- procedural_order_transfer: 28
- failure_signature_transfer: 27
- patch_shape_transfer: 27
- localization_hint: 26
- same_file_transfer: 22
- test_command_transfer: 14
- raw_redundancy_helped: 6
- memory_compression_helped: 3
- same_directory_transfer: 3
- ADP_structure_helped: 3
- raw_noise_hurt: 1

Among raw-only solves:
- localization_hint: 6
- failure_signature_transfer: 6
- patch_shape_transfer: 6
- procedural_order_transfer: 6
- raw_redundancy_helped: 6
- same_file_transfer: 5
- test_command_transfer: 4
- same_directory_transfer: 1

Among memory-only solves:
- same_file_transfer: 3
- localization_hint: 3
- failure_signature_transfer: 3
- patch_shape_transfer: 3
- procedural_order_transfer: 3
- memory_compression_helped: 3
- test_command_transfer: 1
- raw_noise_hurt: 1

Among ADP-only solves:
- localization_hint: 3
- procedural_order_transfer: 3
- ADP_structure_helped: 3
- same_file_transfer: 2
- failure_signature_transfer: 2
- patch_shape_transfer: 2
- test_command_transfer: 1

## Time To Gold File

| condition | resolved | n | avg_first_gold_file_read_step |
| --- | --- | --- | --- |
| no_memory | 0 | 85 | 8.04 |
| no_memory | 1 | 10 | 4.5 |
| raw | 0 | 76 | 5.41 |
| raw | 1 | 19 | 3.05 |
| adp | 0 | 80 | 4.88 |
| adp | 1 | 15 | 3.73 |
| memory | 0 | 79 | 5.65 |
| memory | 1 | 16 | 4.88 |

## Tool Patterns By Outcome

| condition | resolved | n_runs | avg_num_search_commands | avg_num_test_commands | avg_num_edit_commands | rate_patch_touches_gold_file |
| --- | --- | --- | --- | --- | --- | --- |
| adp | 0 | 80 | 7.537 | 4.1 | 4.812 | 0.525 |
| adp | 1 | 15 | 7.4 | 4.867 | 2.8 | 0.867 |
| memory | 0 | 79 | 6.443 | 4.013 | 3.481 | 0.468 |
| memory | 1 | 16 | 12.375 | 4.0 | 5.375 | 1.0 |
| no_memory | 0 | 85 | 10.118 | 3.741 | 4.235 | 0.435 |
| no_memory | 1 | 10 | 15.4 | 4.3 | 2.6 | 0.9 |
| raw | 0 | 76 | 6.671 | 3.855 | 4.263 | 0.566 |
| raw | 1 | 19 | 6.368 | 3.842 | 3.158 | 1.0 |

## Patch Touch Rates

| condition | rate_patch_touches_gold_file | rate_patch_touches_gold_dir |
| --- | --- | --- |
| no_memory | 0.484 | 0.537 |
| raw | 0.653 | 0.726 |
| adp | 0.579 | 0.642 |
| memory | 0.558 | 0.632 |

Successful runs that touched a prior-inspected or prior-edited file: 0.967.

## Representation-Specific Examples

Examples of evidence present in raw but absent or reduced in memory:
- astropy__astropy-15082: raw files/tests include `astropy/nddata/mixins/__init__.py|astropy/nddata/mixins/ndio.py|astropy/nddata/mixins/tests/__init__.py|astropy/nddata/mixins/tests/test_ndio.py|astropy/nddata/mixins/tests/test...`; memory files/tests include `astropy/nddata/__init__.py|astropy/nddata/nddata_withmixins.py|astropy/nddata/mixins/ndarithmetic.py|astropy/nddata/m...`.
- django__django-1891: raw files/tests include `tests/validation/tests.py|tests/validation/models.py|tests/schema/tests.py|tests/schema/fields.py|tests/queryset_pickle/models.py|tests/or_lookups/tests.py|tests/model_forms/tes...`; memory files/tests include `django/forms/models.py|tests/model_forms/tests.py|tests/model_forms/models.py|django/db/models/query.py|tests/runtest...`.
- django__django-30931: raw files/tests include `django/contrib/admin/templatetags/admin_list.py|django/contrib/admin/options.py|django/contrib/admin/checks.py|django/contrib/admin/helpers.py|django/contrib/admin/filters.py|dj...`; memory files/tests include `django/db/models/fields/__init__.py|django/db/models/base.py|tests/model_fields/tests.py|../test_issue.py|django/cont...`.
- matplotlib__matplotlib-25352: raw files/tests include `lib/matplotlib/figure.py|lib/matplotlib/cbook.py|test_pickle_issue.py|opt/anaconda3/lib/python3.8/site.py|lib/matplotlib/__init__.py|test_grouper_pickle.py|test_grouper_minimal....`; memory files/tests include `lib/matplotlib/figure.py|lib/matplotlib/cbook.py|test_grouper_pickle.py|test_grouper_minimal.py|test_grouper_debug.py...`.
- matplotlib__matplotlib-26331: raw files/tests include `/home/jonasg/stuff/bugreport_mpl_toolkits_AxesGrid.py|/home/jonasg/miniconda3/envs/pya/lib/python3.7/site-packages/mpl_toolkits/axes_grid1/axes_grid.py|lib/mpl_toolkits/axes_gri...`; memory files/tests include `/home/jonasg/stuff/bugreport_mpl_toolkits_AxesGrid.py|/home/jonasg/miniconda3/envs/pya/lib/python3.7/site-packages/mp...`.
- sympy__sympy-12426: raw files/tests include `../test_issue.py|opt/anaconda3/lib/python3.8/site.py|sympy/core/basic.py|sympy/solvers/diophantine.py|sympy/plotting/plot.py|sympy/assumptions/sathandlers.py|sympy/matrices/benc...`; memory files/tests include `sympy/matrices/expressions/matexpr.py|sympy/concrete/summations.py|../test_issue.py|sympy/concrete/delta.py|sympy/con...`.

Examples where memory compression appears helpful:
- django__django-31181: memory excerpt `prior_files_inspected: django/contrib/admin/helpers.py|tests/admin_views/customadmin.py|tests/admin_views/admin.py|tests/admin_utils/test_logentry.py|test_issue.py|test_readonly_field_custom_admin.py|django/contrib/admin/models.py|tests/admin_views/tests.py...`.
- django__django-33871: memory excerpt `prior_files_inspected: django/forms/fields.py|django/forms/boundfield.py|django/db/models/fields/__init__.py|django/forms/models.py|django/forms/forms.py|Desktop/swebbed/django/forms/boundfield.py|tests/model_forms/tests.py|../test_issue.py|test_issue.py|te...`.
- django__django-34570: memory excerpt `prior_files_inspected: django/db/models/query.py|django/db/models/sql/query.py|setup.cfg|setup.py || prior_files_edited: django/db/models/sql/query.py || error_failure_assertion_lines: - Traceback (most recent call last):|1736- raise TypeError("Cannot call ...`.

Examples where ADP structure appears helpful:
- scikit-learn__scikit-learn-15093: ADP excerpt ````python|```python-traceback|[ADP ACTION current=1 original=3: api_action Grep]|[ADP ACTION current=2 original=5: api_action Glob]|find . -name "validation.py" -type f | head -20|find testbed -name "validation.py" -type f | grep -E "sklearn|utils|name: Bas...`.
- sympy__sympy-19235: ADP excerpt `[ADP ACTION current=1 original=1: api_action Read]|[ADP ACTION current=2 original=3: api_action Glob]|find . -type d -name "testbed" 2>/dev/null | head -5|[ADP ACTION current=4 original=21: api_action Read]|name: Read:toolu_01KGS3LpzbTufAMCEEtAevxX|[ADP ACT...`.
- sympy__sympy-19484: ADP excerpt `cd ./swebench_9_15/testbed && find . -type f -name "*.py" | head -20|cd testbed && find . -name "*.py" -type f -exec grep -l "class sign" {} /; | head -10|name: Bash:toolu_01RpAdUEhnCN6EzJM7XDVWKY|cd testbed && grep -r "def _eval_rewrite" --include="*.py" |...`.

Examples where prior context is consistent with distraction:
- matplotlib__matplotlib-22482: no_memory solved while prior conditions failed; loser_empty_patch=0, loser_failed=1.

## Paper Framing

The prior-context oracle shows representation-choice headroom over any fixed representation. The transcript evidence is most consistent with localization hints, test/failure transfer, patch-shape transfer, and representation-specific noise or compression effects. These logs support mechanism attribution hypotheses; they do not prove causal mechanisms without ablation.
