# Overlap Results

Official gold patches were available and used.

|bucket_type|bucket|condition|n_targets|resolved_count|success_rate|non_empty_patch_count|patch_attempt_rate|mean_runtime|mean_total_tokens|
|---|---|---|---|---|---|---|---|---|---|
|overlap_bucket|no_gold_file_overlap|no_memory|17|2|0.1176|9|0.5294|2329|2.091e+06|
|overlap_bucket|no_gold_file_overlap|raw|17|3|0.1765|15|0.8824|1970|1.975e+06|
|overlap_bucket|no_gold_file_overlap|adp|17|2|0.1176|14|0.8235|2923|2.611e+06|
|overlap_bucket|no_gold_file_overlap|memory|17|1|0.05882|15|0.8824|2397|2.626e+06|
|overlap_bucket|same_directory_only|no_memory|15|1|0.06667|9|0.6|703.9|9.901e+05|
|overlap_bucket|same_directory_only|raw|15|3|0.2|14|0.9333|949.9|1.701e+06|
|overlap_bucket|same_directory_only|adp|15|1|0.06667|15|1|1095|2.041e+06|
|overlap_bucket|same_directory_only|memory|15|1|0.06667|12|0.8|749.3|1.277e+06|
|overlap_bucket|same_file_overlap|no_memory|63|7|0.1111|44|0.6984|1786|1.641e+06|
|overlap_bucket|same_file_overlap|raw|63|13|0.2063|53|0.8413|1234|1.57e+06|
|overlap_bucket|same_file_overlap|adp|63|12|0.1905|52|0.8254|1539|1.699e+06|
|overlap_bucket|same_file_overlap|memory|63|14|0.2222|52|0.8254|1527|1.394e+06|
|localization_bucket|prior_trajectory_edited_target_gold_file|no_memory|38|3|0.07895|26|0.6842|2095|1.668e+06|
|localization_bucket|prior_trajectory_edited_target_gold_file|raw|38|6|0.1579|33|0.8684|1146|1.516e+06|
|localization_bucket|prior_trajectory_edited_target_gold_file|adp|38|5|0.1316|33|0.8684|1750|1.493e+06|
|localization_bucket|prior_trajectory_edited_target_gold_file|memory|38|5|0.1316|33|0.8684|1587|1.252e+06|
|localization_bucket|prior_trajectory_inspected_target_gold_file|no_memory|33|4|0.1212|23|0.697|1322|1.529e+06|
|localization_bucket|prior_trajectory_inspected_target_gold_file|raw|33|9|0.2727|27|0.8182|1242|1.708e+06|
|localization_bucket|prior_trajectory_inspected_target_gold_file|adp|33|7|0.2121|27|0.8182|1146|2.058e+06|
|localization_bucket|prior_trajectory_inspected_target_gold_file|memory|33|8|0.2424|26|0.7879|1167|1.492e+06|
|localization_bucket|prior_trajectory_never_touched_target_gold_area|no_memory|11|1|0.09091|5|0.4545|2546|2.375e+06|
|localization_bucket|prior_trajectory_never_touched_target_gold_area|raw|11|1|0.09091|9|0.8182|2394|2.215e+06|
|localization_bucket|prior_trajectory_never_touched_target_gold_area|adp|11|1|0.09091|9|0.8182|3641|2.94e+06|
|localization_bucket|prior_trajectory_never_touched_target_gold_area|memory|11|0|0|9|0.8182|2756|3.056e+06|
|localization_bucket|prior_trajectory_same_directory_as_target_gold|no_memory|13|2|0.1538|8|0.6154|881.7|1.066e+06|
|localization_bucket|prior_trajectory_same_directory_as_target_gold|raw|13|3|0.2308|13|1|1123|1.511e+06|
|localization_bucket|prior_trajectory_same_directory_as_target_gold|adp|13|2|0.1538|12|0.9231|1436|1.928e+06|
|localization_bucket|prior_trajectory_same_directory_as_target_gold|memory|13|3|0.2308|11|0.8462|1470|1.629e+06|
