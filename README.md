# Benchmarking Metric-Learning Algorithms

## See this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1kiJ5rKmneQvnYKpVO9vBFdMDNx-yLcXV2wbDXlb-SB8/edit?usp=sharing) for benchmark results (in progress)

## Dependencies
- Python 3.7
- pytorch
- torchvision
- scikit-learn
- faiss-gpu
- tensorboard
- easy_module_attribute_getter
- matplotlib
- pretrainedmodels
- pytorch_metric_learning
- record_keeper
For conda users, package dependencies are also provided in [conda_env.yaml](conda_env.yaml) and [conda_env_all_packages.yaml](conda_env_all_packages.yaml)  

## Organize the datasets (after downloading them)
```
<dataset_root>
|-cub2011
  |-attributes.txt
  |-CUB_200_2011
    |-images
|-cars196
  |-cars_annos.mat
  |-car_ims
|-Stanford_Online_Products
  |-bicycle_final
  |-cabinet_final
  ...
```
Download the datasets here:
- [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [Cars196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [Stanford Online Products](http://cvgl.stanford.edu/projects/lifted_struct)

## Set file paths
1. Open ```configs/config_general/default.yaml```:
    - set ```pytorch_home``` to where you want to save downloaded pretrained models.
    - set ```dataset_root``` to where your datasets are located. 
2. Open run.py:
    - Set the default value for the ```--root_experiment_folder``` flag to where you want all experiment data to be saved.

## Try a basic command
The following command will run an experiment using the default config files in the configs folder.
```
python run.py --experiment_name test1 
```
Experiment data is saved in the following format:
```
<root_experiment_folder>
|-<experiment_name>
  |-configs
    |-config_eval.yaml
    |-config_general.yaml
    |-config_loss_and_miners.yaml
    |-config_models.yaml
    |-config_optimizers.yaml
    |-config_transforms.yaml
  |-<split scheme name>
    |-saved_models
    |-saved_pkls
    |-tensorboard_logs
```
To view experiment data, go to the parent of ```root_experiment_folder``` and start tensorboard: 
```
tensorboard --logdir <root_experiment_folder>
```
Then in your browser, go to ```localhost:6006``` and to see loss histories, optimizer learning rates, train/val accuracy etc. All experiment data is also automatically saved in pickle and CSV format in the ```saved_pkls``` folder.

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
