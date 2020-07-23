# Yaml Syntax

Config files in this library are yaml files, so they follow standard yaml syntax. But you can also use all of the [command line syntax](cl_syntax.md) within your config files. For example, here are two config files in the ```config_general``` category: 

default.yaml
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

num_epochs_train: 1000
save_interval: 2
patience: 9

check_untrained_accuracy: True
skip_eval_if_already_done: True
skip_ensemble_eval_if_already_done: True
save_figures_on_tensorboard: False
save_lists_in_db: False
override_required_compatible_factories: False
```

with_daml.yaml
```yaml
trainer~SWAP~1:
  DeepAdversarialMetricLearning:


trainer~APPLY~2:
  g_alone_epochs: 0
  metric_alone_epochs: 0
  g_triplets_per_anchor: 100
  loss_weights:
    metric_loss: 1
    synth_loss: 0.1
    g_adv_loss: 0.1
    g_hard_loss: 0.1
    g_reg_loss: 0.1
```

The ```with_daml``` config file contains the special flags ```~SWAP~``` and ```~APPLY~```. This particular config file won't work by itself. However, it can be loaded in conjunction with ```default``` at the command line:

```bash
--config_general [default, with_daml]
```

This loads ```default.yaml```, followed by ```with_daml.yaml```. Now the special ```~SWAP~``` and ```~APPLY~``` flags will have an effect. Specifically, ```MetricLossOnly``` will get swapped out for ```DeepAdversarialMetricLearning```, and then the parameters for ```DeepAdversarialMetricLearning``` will be applied to the ```trainer``` dictionary. The final config file ends up looking like this:

```yaml
trainer:
  DeepAdversarialMetricLearning:
    iterations_per_epoch: 100
    dataloader_num_workers: 2
    batch_size: 32
    freeze_trunk_batchnorm: True
    label_hierarchy_level: 0
    loss_weights:
      metric_loss: 1
      synth_loss: 0.1
      g_adv_loss: 0.1
      g_hard_loss: 0.1
      g_reg_loss: 0.1
    set_min_label_to_zero: True
    g_alone_epochs: 0
    metric_alone_epochs: 0
    g_triplets_per_anchor: 100

num_epochs_train: 1000
save_interval: 2
patience: 9

check_untrained_accuracy: True
skip_eval_if_already_done: True
skip_ensemble_eval_if_already_done: True
save_figures_on_tensorboard: False
save_lists_in_db: False
override_required_compatible_factories: False
```


