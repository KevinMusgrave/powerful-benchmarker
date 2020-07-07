# config_dataset

## dataset
The dataset class and its parameters (if any).

Default yaml:
```yaml
dataset:
	CUB200:
```
Command line:
```bash
--dataset {CUB200: {}}
```

## splits_to_eval
The names of splits for which accuracy should be computed.

Default yaml:
```yaml
splits_to_eval:
	- val
```
Command line:
```bash
--splits_to_eval [val]
```

## split_manager
The split manager class and its parameters (if any).

Default yaml:
```yaml
split_manager:
  ClassDisjointSplitManager:
    test_size: 0.5
    test_start_idx: 0.5
    num_training_partitions: 4
    num_training_sets: 4
    hierarchy_level: 0
    data_and_label_getter_keys: [data, label]
```

Command line:
```bash
--split_manager {ClassDisjointSplitManager: {test_size: 0.5, test_start_idx: 0.5, num_training_partitions: 4, num_training_sets: 4, hierarchy_level: 0, data_and_label_getter_keys: [data, label]}}
```