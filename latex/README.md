## powerful-benchmarker/latex

The following commands reads the dataframes created by `../validator_tests/eval_validators.py`, and creates latex tables that summarize the results.

The `--exp_group_excludes_select_best mnist` flag tells the code to ignore `mnist` when determining the best validator per algorithm.

The `--nlargest 5` flag refers to the top-5 checkpoints for computing accuracy.

```
python latex/create_tables.py --exp_group_excludes domainnet126 --exp_group_excludes_select_best mnist --nlargest 5
python latex/create_tables.py --exp_group_prefix domainnet126 --exp_group_excludes_select_best mnist --nlargest 5
python latex/create_tables.py --exp_group_prefix mnist --exp_group_excludes_select_best mnist --nlargest 5
python latex/create_tables.py --exp_group_prefix office --exp_group_excludes_select_best mnist --nlargest 5
python latex/create_tables.py --exp_group_excludes mnist --exp_group_excludes_select_best mnist --nlargest 5
```

Make the `best_accuracy_per_adapter_ranked_by_score` tables use the same color tags as the `best_accuracy_per_adapter` tables.
```
python latex/replace_color_map_tags.py
```

Replace the header for domainnet:
```
python latex/replace_header_acronyms.py
```


### File summary


|Filename|Description|
|-|-|
|`best_accuracy_per_adapter.py`| Best accuracy per adapter per task when using an oracle |
|`best_validator_per_adapter_task.py`| Best validator per algorithm per task |
|`color_map_tags.py`| Code for colorizing latex table cells |
|`correlation_bar_plot_adapter_validator_pairs.py`| Bar plot showing the correlation of every algorithm/validator pair |
|`correlation_bar_plot_single_adapter.py`| Bar plot showing correlation of validators for a single algorithm |
|`correlation_bar_plot.py`| Bar plot showing correlation of validators when applied to all algorithms simultaneously |
|`correlation_diffs.py`| CSV of validator/task pairs with the largest diff between spearman and weighted spearman |
|`correlation_single_adapter.py`| Table showing correlation of validators for a single algorithm |
|`correlation.py`| Table showing correlation of validators when applied to all algorithms simultaneously |
|`create_tables.py`| The main file which calls all other functions |
|`pred_acc_using_best_adapter_validator_pairs.py`| Best accuracy per adapter per task when using the best validator for that algorithm |
|`validator_parameter_explanations.py`| Table explaining validator parameters |