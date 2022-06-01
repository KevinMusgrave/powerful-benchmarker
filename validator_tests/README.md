## powerful-benchmarker/validator_tests


### Order of operations

1. main.py (or run_validators.py) to compute scores
2. collect_dfs.py to gather all dataframe pkls into one dataframe pkl
3. process_df.py
4. per_src_threshold.py


### Common command line flags
The following flags allow for filtering of experiment groups:
| Command-line argument | Description |
| - | - |
|`--exp_groups` | A space delimited list of experiment group names.
|`--exp_group_prefix` | Matches all experiment groups that start with this.
|`--exp_group_suffix` | Matches all experiment groups that end with this.
|`--exp_group_includes` | Matches all experiment groups that have this in their name.
|`--exp_group_excludes` | Matches all experiment groups that do not have this in their name.

These flags are available in the following scripts:

- collect_dfs.py
- create_plots.py
- create_tables.py
- delete_pkls.py
- per_src_threshold.py
- run_validators.py

---
### main.py

This runs a single validator configuration on a single experiment's trials. For example, the following command will compute micro-averaged accuracy on the source validation set for all trials within `<exp_folder>/mnist_mnist_mnistm_fl6_Adam_lr1/dann`:

```
python validator_tests/main.py --exp_group mnist_mnist_mnistm_fl6_Adam_lr1 --exp_name dann \
--validator Accuracy --average=micro --split=src_val
```

---
### run_validators.py

This uses slurm to run a single validator with all of its configurations on multiple experiments. For example, the following will compute source and target accuracies on all 100 trials of the atdoc, dann, and mcc experiments. It will run 4 config/experiment combinations per slurm job:

```
python validator_tests/run_validators.py --slurm_config a100 --run --exp_names atdoc dann mcc --flags Accuracy \
--exp_per_slurm_job 4 --trials_per_exp 100
```