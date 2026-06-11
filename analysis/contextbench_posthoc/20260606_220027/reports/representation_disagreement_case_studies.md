# Representation Disagreement Case Studies

## Case Counts

- exactly_one_prior_representation_solved: 12
- raw_solved_adp_failed: 10
- at_least_two_prior_representations_solved_but_not_all: 10
- raw_solved_memory_failed: 9
- memory_solved_raw_failed: 6
- adp_solved_raw_failed: 6
- no_memory_solved_all_prior_failed: 1
- all_prior_solved_no_memory_failed: 1

## Cases

### astropy__astropy-15082

- Types: raw_solved_memory_failed;raw_solved_adp_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:0,raw:1,adp:0,memory:0
- Winning conditions: raw
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=1, test_reuse=0, failure_reuse=1, loser_empty_patch=1, loser_failed=0
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/astropy__astropy-15082/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/astropy__astropy-15082/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/astropy__astropy-15082/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/astropy__astropy-15082/memory`

### django__django-1891

- Types: raw_solved_memory_failed;raw_solved_adp_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:0,raw:1,adp:0,memory:0
- Winning conditions: raw
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=1, failure_reuse=1, loser_empty_patch=0, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-1891/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-1891/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-1891/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-1891/memory`

### django__django-20098

- Types: raw_solved_adp_failed;at_least_two_prior_representations_solved_but_not_all
- Outcomes: no_memory:0,raw:1,adp:0,memory:1
- Winning conditions: raw;memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=1, failure_reuse=1, loser_empty_patch=1, loser_failed=0
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-20098/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-20098/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-20098/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-20098/memory`

### django__django-28211

- Types: memory_solved_raw_failed;adp_solved_raw_failed;at_least_two_prior_representations_solved_but_not_all
- Outcomes: no_memory:0,raw:0,adp:1,memory:1
- Winning conditions: adp;memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=1, failure_reuse=1, loser_empty_patch=0, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-28211/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-28211/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-28211/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-28211/memory`

### django__django-30254

- Types: raw_solved_memory_failed;at_least_two_prior_representations_solved_but_not_all
- Outcomes: no_memory:0,raw:1,adp:1,memory:0
- Winning conditions: raw;adp
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=1, failure_reuse=1, loser_empty_patch=1, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-30254/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-30254/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-30254/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-30254/memory`

### django__django-30931

- Types: raw_solved_memory_failed;raw_solved_adp_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:0,raw:1,adp:0,memory:0
- Winning conditions: raw
- Evidence in winner missing from loser: contains_relevant_test_command
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=1, failure_reuse=1, loser_empty_patch=1, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-30931/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-30931/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-30931/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-30931/memory`

### django__django-31181

- Types: memory_solved_raw_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:0,raw:0,adp:0,memory:1
- Winning conditions: memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=0, failure_reuse=1, loser_empty_patch=0, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-31181/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-31181/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-31181/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-31181/memory`

### django__django-33461

- Types: raw_solved_memory_failed;at_least_two_prior_representations_solved_but_not_all
- Outcomes: no_memory:1,raw:1,adp:1,memory:0
- Winning conditions: no_memory;raw;adp
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=1, winner_edited_gold=0, test_reuse=1, failure_reuse=1, loser_empty_patch=0, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-33461/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-33461/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-33461/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-33461/memory`

### django__django-33871

- Types: memory_solved_raw_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:0,raw:0,adp:0,memory:1
- Winning conditions: memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=1, test_reuse=1, failure_reuse=1, loser_empty_patch=1, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-33871/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-33871/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-33871/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-33871/memory`

### django__django-34570

- Types: memory_solved_raw_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:0,raw:0,adp:0,memory:1
- Winning conditions: memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=1, winner_edited_gold=1, test_reuse=0, failure_reuse=1, loser_empty_patch=1, loser_failed=0
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-34570/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-34570/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-34570/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/django__django-34570/memory`

### matplotlib__matplotlib-22482

- Types: no_memory_solved_all_prior_failed
- Outcomes: no_memory:1,raw:0,adp:0,memory:0
- Winning conditions: no_memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=0, failure_reuse=0, loser_empty_patch=0, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-22482/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-22482/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-22482/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-22482/memory`

### matplotlib__matplotlib-25352

- Types: raw_solved_memory_failed;raw_solved_adp_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:0,raw:1,adp:0,memory:0
- Winning conditions: raw
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=0, failure_reuse=1, loser_empty_patch=1, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-25352/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-25352/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-25352/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-25352/memory`

### matplotlib__matplotlib-26331

- Types: raw_solved_memory_failed;raw_solved_adp_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:0,raw:1,adp:0,memory:0
- Winning conditions: raw
- Evidence in winner missing from loser: contains_relevant_test_command
- Behavior: winner_earlier_gold=0, winner_edited_gold=1, test_reuse=1, failure_reuse=1, loser_empty_patch=1, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-26331/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-26331/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-26331/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-26331/memory`

### matplotlib__matplotlib-27361

- Types: memory_solved_raw_failed;adp_solved_raw_failed;at_least_two_prior_representations_solved_but_not_all
- Outcomes: no_memory:0,raw:0,adp:1,memory:1
- Winning conditions: adp;memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=0, failure_reuse=1, loser_empty_patch=1, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-27361/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-27361/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-27361/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/matplotlib__matplotlib-27361/memory`

### mwaskom__seaborn-3091

- Types: all_prior_solved_no_memory_failed
- Outcomes: no_memory:0,raw:1,adp:1,memory:1
- Winning conditions: raw;adp;memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=1, winner_edited_gold=1, test_reuse=1, failure_reuse=1, loser_empty_patch=1, loser_failed=0
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/mwaskom__seaborn-3091/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/mwaskom__seaborn-3091/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/mwaskom__seaborn-3091/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/mwaskom__seaborn-3091/memory`

### pylint-dev__pylint-6471

- Types: raw_solved_memory_failed;at_least_two_prior_representations_solved_but_not_all
- Outcomes: no_memory:0,raw:1,adp:1,memory:0
- Winning conditions: raw;adp
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=1, failure_reuse=1, loser_empty_patch=1, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/pylint-dev__pylint-6471/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/pylint-dev__pylint-6471/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/pylint-dev__pylint-6471/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/pylint-dev__pylint-6471/memory`

### scikit-learn__scikit-learn-15093

- Types: adp_solved_raw_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:0,raw:0,adp:1,memory:0
- Winning conditions: adp
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=1, winner_edited_gold=0, test_reuse=0, failure_reuse=1, loser_empty_patch=0, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-15093/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-15093/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-15093/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-15093/memory`

### scikit-learn__scikit-learn-25365

- Types: raw_solved_adp_failed;at_least_two_prior_representations_solved_but_not_all
- Outcomes: no_memory:1,raw:1,adp:0,memory:1
- Winning conditions: no_memory;raw;memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=1, test_reuse=0, failure_reuse=1, loser_empty_patch=1, loser_failed=0
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-25365/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-25365/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-25365/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-25365/memory`

### scikit-learn__scikit-learn-25763

- Types: memory_solved_raw_failed;adp_solved_raw_failed;at_least_two_prior_representations_solved_but_not_all
- Outcomes: no_memory:0,raw:0,adp:1,memory:1
- Winning conditions: adp;memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=1, test_reuse=0, failure_reuse=1, loser_empty_patch=0, loser_failed=0
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-25763/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-25763/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-25763/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/scikit-learn__scikit-learn-25763/memory`

### sympy__sympy-11286

- Types: raw_solved_adp_failed;at_least_two_prior_representations_solved_but_not_all
- Outcomes: no_memory:0,raw:1,adp:0,memory:1
- Winning conditions: raw;memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=1, failure_reuse=1, loser_empty_patch=1, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-11286/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-11286/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-11286/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-11286/memory`

### sympy__sympy-12426

- Types: raw_solved_memory_failed;raw_solved_adp_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:0,raw:1,adp:0,memory:0
- Winning conditions: raw
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=1, failure_reuse=1, loser_empty_patch=0, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-12426/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-12426/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-12426/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-12426/memory`

### sympy__sympy-16689

- Types: raw_solved_adp_failed;at_least_two_prior_representations_solved_but_not_all
- Outcomes: no_memory:1,raw:1,adp:0,memory:1
- Winning conditions: no_memory;raw;memory
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=1, winner_edited_gold=1, test_reuse=0, failure_reuse=1, loser_empty_patch=0, loser_failed=0
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-16689/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-16689/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-16689/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-16689/memory`

### sympy__sympy-19235

- Types: adp_solved_raw_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:1,raw:0,adp:1,memory:0
- Winning conditions: no_memory;adp
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=0, winner_edited_gold=0, test_reuse=0, failure_reuse=0, loser_empty_patch=0, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-19235/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-19235/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-19235/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-19235/memory`

### sympy__sympy-19484

- Types: adp_solved_raw_failed;exactly_one_prior_representation_solved
- Outcomes: no_memory:0,raw:0,adp:1,memory:0
- Winning conditions: adp
- Evidence in winner missing from loser: none
- Behavior: winner_earlier_gold=1, winner_edited_gold=0, test_reuse=1, failure_reuse=1, loser_empty_patch=1, loser_failed=1
- Artifacts: `/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-19484/no_memory|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-19484/raw|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-19484/adp|/home/shashwatsuthar/research/openhands-adp-memory/analysis/contextbench_posthoc/20260606_220027/remote/execution_full_qwen36_65k_fix1_minimal/sympy__sympy-19484/memory`
