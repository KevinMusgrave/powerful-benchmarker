# config_dataset

## dataset
This is the dataset that will be used for training, validation, and testing.

Default yaml:
```yaml
dataset:
	CUB200:
```
Example command line modification:
```bash
# Change dataset to Cars196
--dataset~OVERRIDE~ {Cars196: {}}
```

## splits_to_eval
The names of splits for which accuracy should be computed.

Default yaml:
```yaml
splits_to_eval:
	- val
```
Example command line modification:
```bash
# Eval on train, val, and test.
--splits_to_eval [train, val, test]
```

## split_manager
The split manager determines how the train/val/test splits are formed.

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

Example command line modification:
```bash
# Change number of training sets to 2, and the test size to 0.3
--split_manager~APPLY~2 {test_size: 0.3, num_training_sets: 2}
```