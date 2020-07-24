# Command Line Syntax

This library comes with a powerful command line syntax that makes it easy to change complex configuration options in a precise fashion.

## Lists and dictionaries
Lists and dictionaries are written at the command line in python form:

Example list:
```bash
--splits_to_eval [train, val, test]
```

Example nested dictionary
```bash
--mining_funcs {tuple_miner: {MultiSimilarityMiner: {epsilon: 0.1}}}
```


## Merge
Consider the following optimizer configuration.

```yaml
optimizers:
  trunk_optimizer:
    RMSprop:
      lr: 0.000001
```

At the command line, we can change ```lr``` to 0.01, and add ```alpha = 0.95``` to the ```RMSprop``` parameters:
```bash
--optimizers {trunk_optimizer: {RMSprop: {lr: 0.01, alpha: 0.95}}}
```
So in effect, the config file now looks like this:
```yaml
optimizers:
  trunk_optimizer:
    RMSprop:
      lr: 0.01
      alpha: 0.95
```
In other words, we specify a dictionary at the command line, using python dictionary syntax. This dictionary is then merged into the one specified in the config file. Thus, adding keys is very straightforward:
```bash
--optimizers {embedder_optimizer: {Adam: {lr: 0.01}}}
```
Now the config file includes a specification for ```embedder_optimizer```:
```yaml
optimizers:
  trunk_optimizer:
    RMSprop:
      lr: 0.000001
  embedder_optimizer:
    Adam:
      lr: 0.01
```

But what happens if we try to set the ```trunk_optimizer``` to ```Adam```?
```bash
--optimizers {trunk_optimizer: {Adam: {lr: 0.01}}}
```
Now there's a problem with the config file, because two optimizer types are specified for a single optimizer:
```yaml
optimizers:
  trunk_optimizer:
    RMSprop:
      lr: 0.000001
    Adam:
      lr: 0.01
```

How can we get around this? By using the [Override](cl_syntax.md#override) syntax.

## Override

Overriding simple options requires no special syntax. For example, the following will change ```save_interval``` from its default value of 2 to 5:
```bash
--save_interval 5
```

However, for complex options (i.e. nested dictionaries) the ```~OVERRIDE~``` flag is required to avoid merges. Let's consider the same optimizer config file from above:
```yaml
optimizers:
  trunk_optimizer:
    RMSprop:
      lr: 0.000001
```
To instead use ```Adam``` with ```lr = 0.01```:
```bash
--optimizers~OVERRIDE~ {trunk_optimizer: {Adam: {lr: 0.01}}}
```
Now the config file looks like this:
```yaml
optimizers:
  trunk_optimizer:
    Adam:
      lr: 0.01
```
The ```~OVERRIDE~``` flag can be used at any level of the dictionary, which comes in handy for more complex config options. Consider this config file:
```yaml
optimizers:
  trunk_optimizer:
    RMSprop:
      lr: 0.000001
  embedder_optimizer:
    Adam:
      lr: 0.01
```
We can make ```trunk_optimizer``` use Adam, but leave ```embedder_optimizer``` unchanged, by applying the ```~OVERRIDE~``` flag to ```trunk_optimizer```:
```bash
--optimizers {trunk_optimizer~OVERRIDE~: {Adam: {lr: 0.01}}} 
```

## Apply
Sometimes the merging and override capabilities don't offer enough flexibility. Consider this config file:
```yaml
trainer:
  MetricLossOnly:
    dataloader_num_workers: 2
    batch_size: 32
```
If we want to change the batch size using merging:
```bash
--trainer: {MetricLossOnly: {batch_size: 256}}
```
There are two problems with this:

1. It's verbose. We only wanted to change ```batch_size```, but we had to write out the name of the ```trainer```.
2. It requires knowledge of the ```trainer``` that is being used.

So instead, we can use the ```~APPLY~``` flag:
```bash
--trainer~APPLY~2: {batch_size: 256}
```
This syntax means that ```{batch_size: 256}``` will be applied to (i.e. merged into) all dictionaries at a depth of 2. So the ```trainer``` config now looks like:
```yaml
trainer:
  MetricLossOnly:
    dataloader_num_workers: 2
    batch_size: 256
```
Here's another example with optimizers. The starting configuration looks like:
```yaml
optimizers:
  trunk_optimizer:
    RMSprop:
      lr: 0.000001
  embedder_optimizer:
    Adam:
      lr: 0.01
```
We can set both learning rates to 0.005:
```bash
--optimizers~APPLY~3 {lr: 0.005}
```
The new config file looks like:
```yaml
optimizers:
  trunk_optimizer:
    RMSprop:
      lr: 0.005
  embedder_optimizer:
    Adam:
      lr: 0.005
```

## Swap
Consider the ```trainer``` config file again, but with more of its parameters listed:
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
Let's say you write your own custom trainer, and it has the same set of initialization parameters. One way to use your custom trainer is to use the ```~OVERRIDE~``` flag:
```bash
--trainer~OVERRIDE~ {YourCustomTrainer: {iterations_per_epoch: 100, \
dataloader_num_workers: 32, \
batch_size: 32, \
freeze_trunk_batchnorm: True, \
label_hierarchy_level: 0, \
loss_weights: null, \
set_min_label_to_zero: True}}
```
Again, this is very verbose, considering we only wanted to change the ```trainer``` type. So instead, we can use the ```~SWAP~``` flag:
```bash
--trainer~SWAP~1 {YourCustomTrainer: {}}
```
This goes to a dictionary depth of 1, and swaps the only key, ```MetricLossOnly```, with ```YourCustomTrainer```, while leaving everything else unchanged. Now the config file looks like:
```yaml
trainer:
  YourCustomTrainer:
    iterations_per_epoch: 100
    dataloader_num_workers: 2
    batch_size: 32
    freeze_trunk_batchnorm: True
    label_hierarchy_level: 0
    loss_weights: null
    set_min_label_to_zero: True
```
What if there are multiple keys at the specified depth? For example, consider this configuration for data transforms:
```yaml
transforms:
  train:
    Resize:
      size: 256
    RandomResizedCrop:
      scale: 0.16 1
      ratio: 0.75 1.33
      size: 227
    RandomHorizontalFlip:
      p: 0.5
```
If we want to swap ```RandomHorizontalFlip``` out for ```RandomVerticalFlip```, we need to explicitly indicate the mapping, because there are 2 other keys that could be swapped out (```Resize``` and ```RandomResizedCrop```):
```bash
--transforms~SWAP~2 {RandomHorizontalFlip: RandomVerticalFlip}
```
The new config file contains ```RandomVerticalFlip``` in place of ```RandomHorizontalFlip```:
```yaml
transforms:
  train:
    Resize:
      size: 256
    RandomResizedCrop:
      scale: 0.16 1
      ratio: 0.75 1.33
      size: 227
    RandomVerticalFlip:
      p: 0.5
```


## Delete

Consider this ```models``` config file:
```yaml
models:
  trunk:
    bninception:
      pretrained: imagenet
  embedder:
    MLP:
      layer_sizes:
        - 128
```
Let's replace ```embedder``` with ```Identity()```, which is a essentially an empty PyTorch module:

```bash
--models {embedder~OVERRIDE~ {Identity: {}}}
```

But because ```embedder``` has no optimizable parameters, we need to get rid of the ```embedder_optimizer``` that is specified in the default config file. We can do this easily with the ```~DELETE~``` flag:
```bash
--optimizers {embedder_optimizer~DELETE~: {}}
```