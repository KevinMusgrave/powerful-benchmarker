## powerful-benchmarker/powerful_benchmarker

### Getting started

#### Training a source-only model

Train a model on MNIST for 20 epochs:
```
python powerful_benchmarker/main.py --exp_name test_experiment0 --dataset mnist \
--src_domains mnist --adapter PretrainerConfig \
--download_datasets --num_trials 1 \
--max_epochs 20 --pretrain_on_src --validator src_accuracy \
--use_stat_getter
```

Test the source-only model on MNIST and MNISTM:
```
python powerful_benchmarker/main.py --exp_name test_experiment0 \
--target_domains mnist mnistm --evaluate --validator oracle
```

#### Hyperparameter search for domain adaptation algorithms

Train a model using DANN for 20 epochs, with 10 different random hyperparameter settings. Save features every 5 epochs.
```
python powerful_benchmarker/main.py --exp_name dann_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter DANNConfig \
--feature_layer 6 \
--max_epochs 20 --num_trials 10 --save_features \
--val_interval 5
```




### main.py
| Command-line argument | Description |
| - | - |
|`--exp_folder` | The root experiment folder. This defaults to the value set in [constants.yaml](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/constants.yaml).
|`--dataset_folder` | Where the datasets exist or where they will be downloaded to. This defaults to the value set in [constants.yaml](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/constants.yaml).
|`--dataset` | The dataset that contains the source and target domains. Choices are `mnist`, `office31`, or `officehome` 
|`--src_domains` | The name of the source domain. For example, if the dataset is `office31`, then this could be one of `amazon`, `dslr`, or `webcam`. 
|`--target_domains` | The name of the target domain. 
|`--adapter` | The name of the domain adaptation algorithm configuration, e.g. `DANNConfig`. See the [configs](https://github.com/KevinMusgrave/powerful-benchmarker/tree/master/powerful_benchmarker/configs) folder for all available configurations. 
|`--exp_name` | Experiment data will be saved in `<exp_folder>/<exp_name>`.
|`--max_epochs` | Training will stop after this many epochs. For domain adaptation, the number of epochs is based on the length of the target domain.
|`--patience` | Training will stop if the validation score does not improve after this many epochs. This only works if `--validator` is specified.
|`--val_interval` | The validation data will be used every `val_interval` epochs. For example, if a `validator` is specified, then the validation score will be computed every `val_interval` epochs.
|`--batch_size` | The amount of data passed to the model per iteration. The batch size is for both source and target domains. For example, a batch size of 64 means 64 source images and 64 target images, for a total of 128.
|`--num_workers` | The number of PyTorch dataloader workers for loading images 
|`--num_trials` | The number of hyperparameter settings to be tried. Each trial gets its own folder: `<exp_folder>/<exp_name>/<trial_num>`.
|`--n_startup_trials` | The number of trials with randomly picked hyperparameters, before a hyperparameter optimization algorithm is used.
|`--validator` | If specified, a validation score will be computed every `val_interval` epochs.
|`--pretrain_on_src` | Add this flag to train a source-only model.
|`--evaluate` | Add this flag to evaluate the best model of an existing experiment.
|`--num_reproduce` | The best hyperparameters will be used to train this many more models. This is useful if you want to get the standard deviation of an algorithm's performance.
|`--feature_layer` | If 0, then the output of the trunk (a.k.a. feature generator) is used as the "features". Higher numbers correspond with layers of the classifier model. For example, if set to 3, then the 3rd layer of the classifier model will be used as features. See the `set_feature_layer` function in [BaseConfig](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/powerful_benchmarker/configs/base_config.py).
|`--optimizer` | Either "SGD" or "Adam".
|`--lr_multiplier` | The base learning rate will be multiplied by this amount for certain models or layers, depending on the adapter config.
|`--pretrain_lr` | The learning rate used for training a source-only model.
|`--fixed_param_source` | Hyperparameters will be loaded from the best trial of `<exp_folder>/<fixed_param_source>`. For example, when trying MCC-DANN, you may want to load the best hyperparameters from the DANN experiment, so that the search space is limited to only the MCC-related hyperparameters.
|`--save_features` | Add this flag to save features every `val_interval` epochs. See [utils/ignite_save_features](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/powerful_benchmarker/utils/ignite_save_features.py) for details.
|`--download_datasets` | Add this flag to automatically download datasets to `dataset_folder` if they aren't already present.
|`--use_stat_getter` | Add this flag to compute source and target accuracies every `val_interval` epochs. This is independent of `validator`.
|`--check_initial_score` | Add this flag to compute a validation score before training begins. This is relevant only if `validator` is specified.
|`--use_full_inference` | Add this flag to retrieve all available model features during each validation step. For example, without this flag, the inference step usually just returns "features" and "logits". But with this flag, it might also return discriminator logits, or the logits from multiple classifiers (it depends on the model architecture). This is particularly relevant if `save_features` is set.


### launch_multiple.py
| Command-line argument | Description |
| - | - |
|--exp_config | The name of the experiment config yaml file. For example, `mnist/mnist_fl3_adam_lr1` refers to [this yaml config file](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/powerful_benchmarker/yaml_configs/exp_configs/mnist/mnist_fl3_adam_lr1.yaml).


### launch_one.py
| Command-line argument | Description |
| - | - |
|`--script_wrapper_timeout` | How many seconds an experiment folder can go unchanged before the experiment is killed and restarted. This can be useful if you have issues with your experiments occasionally hanging.
|`--config_names` | A space delimited list of lowercase adapter names, e.g. `dann mcc`.
|`--slurm_config` | The name of the slurm yaml config file containing slurm-related config options.
|`--group_configs` | A space delimited list of yaml config file names containing experiment settings that ultimately get passed to `main.py`.
|`--src_domains` | Source domain can be set in a group config, or via command line argument here.
|`--target_domains` | Target domain can be set in a group config, or via command line argument here.


### delete_experiment.py
| Command-line argument | Description |
| - | - |
|`--adapter` | The adapter experiment to delete. For example, `dann` will find all folders matching `<exp_folder>/<exp_group>/dann`.
|`--delete` | The found folders will be deleted only if this flag is used.


### delete_exp_groups.py
This will delete all experiments within experiment groups matching the `--exp_group` flag filters.
| Command-line argument | Description |
| - | - |
|`--delete` | The found folders will be deleted only if this flag is used.