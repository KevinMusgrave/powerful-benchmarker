## powerful-benchmarker/powerful_benchmarker

### main.py
| Command-line argument | Description |
| - | - |
|`--exp_folder` | The root experiment folder. This defaults to the value set in [constants.yaml](https://github.com/KevinMusgrave/powerful-benchmarker/blob/dev/constants.yaml).
|`--dataset_folder` | Where the datasets exist or where they will be downloaded to. This defaults to the value set in [constants.yaml](https://github.com/KevinMusgrave/powerful-benchmarker/blob/dev/constants.yaml).
|`--dataset` | The dataset that contains the source and target domains. Choices are `mnist`, `office31`, or `officehome` 
|`--src_domains` | The name of the source domain. For example, if the dataset is `office31`, then this could be one of `amazon`, `dslr`, or `webcam`. 
|`--target_domains` | The name of the target domain. 
|`--adapter` | The name of the domain adaptation algorithm configuration, e.g. `DANNConfig`. See the [configs](https://github.com/KevinMusgrave/powerful-benchmarker/tree/dev/powerful_benchmarker/configs) folder for all available configurations. 
|`--exp_name` | Experiment data will be saved in `<exp_folder>/<exp_name>`.
|`--max_epochs` | Training will stop after this many epochs. For domain adaptation, the number of epochs is based on the length of the target domain.
|`--patience` | Training will stop if the validation score does not improve after this many epochs. This only works if `--validator` is specified.
|`--val_interval` | The validation data will be used every `val_interval` epochs. For example, if a `validator` is specified, then the validation score will be computed every `val_interval` epochs.
|`--batch_size` | The amount of data passed to the model per iteration. The batch size is for both source and target domains. For example, a batch size of 64 means 64 source images and 64 target images, for a total of 128.
|`--num_workers` | The number of PyTorch dataloader workers for loading images 
|`--num_trials` | The number of hyperparameter settings to be tried. Each trial gets its own folder: `<exp_folder>/<exp_name>/<trial_num>`.
|`--n_startup_trials` | The number of trials with randomly picked hyperparameters, before a hyperparameter optimization algorithm is used.
|`--start_with_pretrained` | Add this flag to load the source-only model's weights. Otherwise the model will be randomly initialized.
|`--validator` | If specified, a validation score will be computed every `val_interval` epochs.
|`--pretrain_on_src` | Add this flag to train a source-only model.
|`--evaluate` | Add this flag to evaluate the best model of an existing experiment.
|`--num_reproduce` | The best hyperparameters will be used to train this many more models. This is useful if you want to get the standard deviation of an algorithm's performance.
|`--feature_layer` | If 0, then the output of the trunk (a.k.a. feature generator) is used as the "features". Higher numbers correspond with layers of the classifier model. For example, if set to 3, then the 3rd layer of the classifier model will be used as features. See the `set_feature_layer` function in [BaseConfig](https://github.com/KevinMusgrave/powerful-benchmarker/blob/dev/powerful_benchmarker/configs/base_config.py).
|`--optimizer` | Either "SGD" or "Adam".
|`--lr_multiplier` |
|`--pretrain_lr` |
|`--fixed_param_source` |
|`--save_features` |
|`--download_datasets` |
|`--use_stat_getter` |
|`--check_initial_score` |
|`--use_full_inference` |