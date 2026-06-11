# Paper-Ready Tables

## Condition Summary
|condition|n_targets|resolved_count|success_rate|wilson_95_low|wilson_95_high|non_empty_patch_count|empty_patch_count|
|---|---|---|---|---|---|---|---|
|no_memory|95|10|0.1053|0.05819|0.183|62|33|
|raw|95|19|0.2|0.1319|0.2914|82|13|
|adp|95|15|0.1579|0.09808|0.2443|81|14|
|memory|95|16|0.1684|0.1064|0.2562|79|16|

## Paired McNemar
|condition_a|condition_b|n|a_resolved_count|b_resolved_count|rate_diff_a_minus_b|a_only_resolved|b_only_resolved|mcnemar_exact_p|holm_adjusted_p_across_six_comparisons|
|---|---|---|---|---|---|---|---|---|---|
|raw|no_memory|95|19|10|0.09474|11|2|0.02246|0.1348|
|memory|no_memory|95|16|10|0.06316|9|3|0.146|0.73|
|adp|no_memory|95|15|10|0.05263|8|3|0.2266|0.9062|
|raw|memory|95|19|16|0.03158|9|6|0.6072|1|
|raw|adp|95|19|15|0.04211|10|6|0.4545|1|
|memory|adp|95|16|15|0.01053|7|6|1|1|

## Solve Patterns
|no_memory|raw|adp|memory|count|
|---|---|---|---|---|
|0|0|0|0|66|
|0|1|0|0|6|
|1|1|1|1|5|
|0|0|0|1|3|
|0|0|1|1|3|
|0|0|1|0|2|
|0|1|0|1|2|
|0|1|1|0|2|
|1|1|0|1|2|
|0|1|1|1|1|
|1|0|0|0|1|
|1|0|1|0|1|
|1|1|1|0|1|

## Patch Attempt Summary
|condition|n|empty_patch_count|non_empty_patch_count|patch_attempt_rate|resolved_count|resolved_given_attempt_rate|
|---|---|---|---|---|---|---|
|no_memory|95|33|62|0.6526|10|0.1613|
|raw|95|13|82|0.8632|19|0.2317|
|adp|95|14|81|0.8526|15|0.1852|
|memory|95|16|79|0.8316|16|0.2025|

## Overlap Bucket Results
|bucket_type|bucket|condition|n_targets|resolved_count|success_rate|patch_attempt_rate|
|---|---|---|---|---|---|---|
|overlap_bucket|no_gold_file_overlap|no_memory|17|2|0.1176|0.5294|
|overlap_bucket|no_gold_file_overlap|raw|17|3|0.1765|0.8824|
|overlap_bucket|no_gold_file_overlap|adp|17|2|0.1176|0.8235|
|overlap_bucket|no_gold_file_overlap|memory|17|1|0.05882|0.8824|
|overlap_bucket|same_directory_only|no_memory|15|1|0.06667|0.6|
|overlap_bucket|same_directory_only|raw|15|3|0.2|0.9333|
|overlap_bucket|same_directory_only|adp|15|1|0.06667|1|
|overlap_bucket|same_directory_only|memory|15|1|0.06667|0.8|
|overlap_bucket|same_file_overlap|no_memory|63|7|0.1111|0.6984|
|overlap_bucket|same_file_overlap|raw|63|13|0.2063|0.8413|
|overlap_bucket|same_file_overlap|adp|63|12|0.1905|0.8254|
|overlap_bucket|same_file_overlap|memory|63|14|0.2222|0.8254|
|localization_bucket|prior_trajectory_edited_target_gold_file|no_memory|38|3|0.07895|0.6842|
|localization_bucket|prior_trajectory_edited_target_gold_file|raw|38|6|0.1579|0.8684|
|localization_bucket|prior_trajectory_edited_target_gold_file|adp|38|5|0.1316|0.8684|
|localization_bucket|prior_trajectory_edited_target_gold_file|memory|38|5|0.1316|0.8684|
|localization_bucket|prior_trajectory_inspected_target_gold_file|no_memory|33|4|0.1212|0.697|
|localization_bucket|prior_trajectory_inspected_target_gold_file|raw|33|9|0.2727|0.8182|
|localization_bucket|prior_trajectory_inspected_target_gold_file|adp|33|7|0.2121|0.8182|
|localization_bucket|prior_trajectory_inspected_target_gold_file|memory|33|8|0.2424|0.7879|
|localization_bucket|prior_trajectory_never_touched_target_gold_area|no_memory|11|1|0.09091|0.4545|
|localization_bucket|prior_trajectory_never_touched_target_gold_area|raw|11|1|0.09091|0.8182|
|localization_bucket|prior_trajectory_never_touched_target_gold_area|adp|11|1|0.09091|0.8182|
|localization_bucket|prior_trajectory_never_touched_target_gold_area|memory|11|0|0|0.8182|
|localization_bucket|prior_trajectory_same_directory_as_target_gold|no_memory|13|2|0.1538|0.6154|
|localization_bucket|prior_trajectory_same_directory_as_target_gold|raw|13|3|0.2308|1|
|localization_bucket|prior_trajectory_same_directory_as_target_gold|adp|13|2|0.1538|0.9231|
|localization_bucket|prior_trajectory_same_directory_as_target_gold|memory|13|3|0.2308|0.8462|

## Qualitative Case Index
|case_type|instance_id|repo|condition_outcomes|overlap_bucket|mechanism_hypothesis|
|---|---|---|---|---|---|
|raw_solved_memory_failed|astropy__astropy-15082|astropy/astropy|no_memory:0,raw:1,adp:0,memory:0|same_file_overlap|same_file_transfer|
|memory_solved_raw_failed|django__django-28211|django/django|no_memory:0,raw:0,adp:1,memory:1|same_file_overlap|same_file_transfer|
|raw_solved_adp_failed|django__django-1891|django/django|no_memory:0,raw:1,adp:0,memory:0|same_file_overlap|same_file_transfer|
|adp_solved_raw_failed|matplotlib__matplotlib-27361|matplotlib/matplotlib|no_memory:0,raw:0,adp:1,memory:1|same_file_overlap|same_file_transfer|
|all_prior_solved_no_memory_failed|mwaskom__seaborn-3091|mwaskom/seaborn|no_memory:0,raw:1,adp:1,memory:1|same_file_overlap|same_file_transfer|
|no_memory_solved_all_prior_failed|matplotlib__matplotlib-22482|matplotlib/matplotlib|no_memory:1,raw:0,adp:0,memory:0|same_directory_only|localization_hint|
|empty_no_memory_nonempty_resolved_prior|django__django-30254|django/django|no_memory:0,raw:1,adp:1,memory:0|no_gold_file_overlap|localization_hint|

## Runtime/Token Summary
|condition|total_wall_seconds|mean_wall_seconds|median_wall_seconds|total_input_tokens|total_output_tokens|total_total_tokens|total_llm_calls|
|---|---|---|---|---|---|---|---|
|no_memory|1.627e+05|1713|800.7|1.491e+08|4.678e+06|1.538e+08|4416|
|raw|1.255e+05|1321|746.7|1.547e+08|3.311e+06|1.58e+08|3708|
|adp|1.631e+05|1716|843.3|1.772e+08|4.887e+06|1.82e+08|3542|
|memory|1.482e+05|1560|676.7|1.478e+08|3.844e+06|1.516e+08|3858|
