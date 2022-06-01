## powerful-benchmarker/validator_tests


### Order of operations

1. main.py (or run_validators.py) to compute scores
2. collect_dfs.py to gather all dataframe pkls into one dataframe pkl
3. process_df.py
4. per_src_threshold.py
5. create_tables.py and create_plots.py


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

This runs a single validator configuration on a single experiment's trials. It will save a pkl file containing the validation scores, within each trial's folder. For example, the following command will compute micro-averaged accuracy on the source validation set for all trials within `<exp_folder>/mnist_mnist_mnistm_fl6_Adam_lr1/dann`:

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

See [scripts/run.py](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/validator_tests/scripts/run.py), [scripts/mnist.sh](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/validator_tests/scripts/mnist.sh), [scripts/office31sh](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/validator_tests/scripts/office31.sh), and [scripts/officehome.sh](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/validator_tests/scripts/officehome.sh) for examples.


---
### collect_dfs.py

After computing validation scores and saving them to pkl files, you can gather them into larger files using `collect_dfs.py`. For example, this will gather all pkls under each experiment group that starts with "mnist", and save them as `all_dfs.pkl` within that same experiment group:

```
python collect_dfs.py --exp_group_prefix mnist
```

---
### process_df.py
This makes some modifications to `all_dfs.pkl`, like removing irrelevant column names. The new file will be `all_dfs_processed.pkl`, saved in the same folder as `all_dfs.pkl`:

```
python process_df.py --exp_group_prefix mnist
```

---
### per_src_threshold.py
The next step is to compute relative target accuracies and Spearman correlation at various source validation thresholds.

See [scripts/per_src_threshold.sh](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/validator_tests/scripts/per_src_threshold.sh) for examples.


---
### create_tables.py


---
### create_plots.py
