# config_general

## api_parser

The API parser controls the experiment by making objects and running trainers and testers. (The API parser is called by the Runner, which is the entry point to the program.) 

By default, ```BaseAPIParser``` is used. If you use a custom trainer, the code will try to use ```API<name_of_your_trainer>```, and if that doesn't exist, it will fall back to ```BaseAPIParser```. However, if you explicitly set the ```api_parser``` option, then the one you specify will be used.

Default yaml:
```yaml
api_parser: null
```
Example command line modification:
```bash
--api_parser {YourCustomAPIParser: {}}
```


## trainer
The trainer trains your model.

Default yaml:
```yaml
trainer:
  MetricLossOnly:
    iterations_per_epoch: 100
    dataloader_num_workers: 2
    batch_size: 32
    freeze_trunk_batchnorm: True
    label_hierarchy_level: 0
    loss_weights: null
    set_min_label_to_zero: True
```
Example command line modification:
```bash
# Swap in a different trainer, but keep the input parameters the same
--trainer~SWAP~1 {CascadedEmbeddings: null}
```


## num_epochs_train
The maximum number of epochs to train for.

Default yaml:
```yaml
num_epochs_train: 1000
```

Example command line modification:
```bash
--num_epochs_train 100
```

## save_interval
Models will be evaluated and saved every ```save_interval``` epochs.

Default yaml:
```yaml
save_interval: 2
```

Example command line modification:
```bash
--save_interval 10
```

## patience
Training will end if the validation accuracy stops improving after ```patience+1``` epochs.

Default yaml:
```yaml
save_interval: 2
```

Example command line modification:
```bash
# Don't use patience at all
--patience null
```


## check_untrained_accuracy
If ```True```, then the tester will compute accuracy for the initial trunk (epoch -1) and initial trunk + embedder (epoch 0). Otherwise, these will be skipped.

Default yaml:
```yaml
check_untrained_accuracy: True
```

Example command line modification:
```bash
--check_untrained_accuracy False
```

## skip_eval_if_already_done
If ```True```, then the tester will skip evaluation if a split/epoch has already been logged in the log files. If ```False```, then the tester will evaluate a split/epoch regardless of whether it has already been done in the past. Previous logs will be preserved, hence the logs will contain duplicate results, and the most recent version for any split/epoch will be considered the "official" value for that split/epoch.

Default yaml:
```yaml
skip_eval_if_already_done: True
```

Example command line modification:
```bash
--skip_eval_if_already_done False
```

## skip_ensemble_eval_if_already_done
The same as ```skip_eval_if_already_done```, but for ensembles.

Default yaml:
```yaml
skip_ensemble_eval_if_already_done: True
```

Example command line modification:
```bash
--skip_ensemble_eval_if_already_done False
```

## log_data_to_tensorboard
Set to False if you don't want to log data to tensorboard. You might want to do this if your disk I/O is slow.

Default yaml:
```yaml
log_data_to_tensorboard: True
```

Example command line modification:
```bash
--log_data_to_tensorboard False
```


## save_figures_on_tensorboard
Use matplotlib to plot things on tensorboard. (Most data doesn't require matplotlib.)

Default yaml:
```yaml
save_figures_on_tensorboard: False
```

Example command line modification:
```bash
--save_figures_on_tensorboard True
```

## save_lists_in_db
In record-keeper, non-scalar values are saved in the database as json-lists. This setting is False by default, because these lists can sometimes be quite large, causing the database file size to grow quickly.

Default yaml:
```yaml
save_lists_in_db: False
```

Example command line modification:
```bash
--save_lists_in_db True
```

## override_required_compatible_factories
Each APIParser comes with predefined compatible factories, which are used by default, regardless of what is specified in the ```factories``` config option. This allows you to specify a trainer without having to specify all the required factories. However, if you have your own custom factory that you know is compatible, and want to use that instead, you should set this flag to True.

Default yaml:
```yaml
override_required_compatible_factories: False
```

Example command line modification:
```bash
--override_required_compatible_factories True
```