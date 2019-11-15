# powerful_benchmarker

## See this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1kiJ5rKmneQvnYKpVO9vBFdMDNx-yLcXV2wbDXlb-SB8/edit?usp=sharing) for benchmark results (in progress)

## Dependencies
This library was tested using Python 3.7. For package dependencies see [conda_env.yaml](conda_env.yaml). 

## Basic Usage
First, open configs/config_general/default.yaml, and set pytorch_home and dataset_root.
Next, open run.py and change the default value for the "--root_experiment_folder" flag. This is where you want all experiment data to be saved.

Assuming you have the CUB200 dataset stored at dataset_root, you should be able to run the following command, which will run an experiment using the default config files in the configs folder. The experiment data will be saved in <your_root_experiment_folder>/test1.
```
python run.py --experiment_name test1 
```
To view experiment data, go to the parent folder of your root experiment folder and start tensorboard. 
```
tensorboard --logdir <your_root_experiment_folder> --port=<port_you_want_to_use>
```
Then in your browser, go to localhost:<port_you_want_to_use> and you will see experiment data, like loss histories, optimizer learning rates, train/test accuracy etc. All experiment data is also automatically saved in pickle and CSV format in each experiment folder.

## Config options
### config_general
```yaml
pytorch_home: <Path where you save pretrained pytorch models>
dataset_root: <Path where you keep your datasets>

training_method: <type> #options: MetricLossOnly, TrainWithClassifier, CascadedEmbeddings, DeepAdversarialMetricLearning
testing_method: <type> #options: GlobalEmbeddingSpaceTester, WithSameParentLabelTester
dataset:  
  <type>: #options: CUB200, Cars196, StanfordOnlineProducts
    <kwarg>: <value>
    ...
num_epochs_train: <how long to train for>
iterations_per_epoch: <how long an "epoch" lasts>
save_interval: <how often (in number of epochs) models will be saved and evaluated>
split:
  schemes: <list>
num_variants_per_split_scheme: <number of ways the dataset will be split, per split scheme> 

label_hierarchy_level: <number>
dataloader_num_workers: <number>
skip_eval: <boolean>
check_pretrained_accuracy: <boolean>

```
### config_models
```yaml
models:
  trunk:
    <type>:
      <kwarg>: <value>
      ...
  embedder:
    <type>:
      <kwarg>: <value>
      ...
batch_size: <number>
freeze_batchnorm: <boolean>
```
### config_loss_and_miners
```yaml 
loss_funcs:
  <name>: 
    <type>:
      <kwarg>: <value>
      ...
  ...

sampler:
  <type>:
    <kwarg>: <value>
    ...

mining_funcs:
  <name>: 
    <type>: 
      <kwarg>: <value>
      ...
  ...
```
### config_optimizers
```yaml
optimizers:
  trunk_optimizer:
    <type>:
      <kwarg>: <value>
      ...
  embedder_optimizer:
    <type>:
      <kwarg>: <value>
      ...
  ...
```
### config_transforms
```yaml
transforms:
  train:
    <type>
      <kwarg>: <value>
      ...
    ...

  eval:
    <type>
      <kwarg>: <value>
      ...
    ...
```
### config_eval
```yaml
eval_reference_set: <name> #options: compared_to_self, compared_to_sets_combined, compared_to_training_set
eval_normalize_embeddings: <boolean>
eval_use_trunk_output: <boolean>
eval_batch_size: <number>
eval_metric_for_best_epoch: <name> #options: NMI, recall_at_1, r_precision, ordered_r_precision
eval_dataloader_num_workers: <number>
```
