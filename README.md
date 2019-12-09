# Benchmarking Metric-Learning Algorithms the Right Way

## [Benchmark results (in progress)](https://drive.google.com/open?id=1Y_stkiqlHA7HTMNrhyPCnYhR0oevphRR): 
- [Train/val 50/50](https://docs.google.com/spreadsheets/d/1kiJ5rKmneQvnYKpVO9vBFdMDNx-yLcXV2wbDXlb-SB8/edit?usp=sharing)

## See [pytorch_metric_learning](https://github.com/KevinMusgrave/pytorch_metric_learning) for a list of currently available losses, miners, samplers, training methods, and testing methods.

## Why use this tool?
1. Flexibility and power:
    - Configure most aspects of your experiment easily with config files and/or the command-line. Extend existing config files by merging them with new ones, or by merging/overriding config options via the command line.
    - Mix and match losses, mining functions, samplers, and training methods.
2. Detailed record keeping:
    - View in-depth information about the training process on Tensorboard, and save data in pickle and csv format.
    - View the history (if any) of config options that were changed during the course of an experiment. 
3. Better performance metrics
    - Use metrics that are more informative than Recall@1,2,4,8.
    - Measure accuracy on multiple class-based train/val/test splits.

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

## Set default file paths
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
Then in your browser, go to ```localhost:6006``` to see things like loss histories, optimizer learning rates, and train/val accuracy. The tensorboard also includes other interesting information that can help you understand the training process. For example, taken alone, this contrastive loss history plot makes it look like training never progressed:  

![loss_history_example](readme_imgs/loss_history_example.png)

But take a look at the number of pairs that violate the margin:

![nonzero_pairs_example](readme_imgs/nonzero_pairs_example.png)

And look at the number of hard postive and hard negative pairs that the miner is able to extract:

![miner_info_example](readme_imgs/miner_info_example.png)

To learn more about where this info comes from, check out [pytorch_metric_learning](https://github.com/KevinMusgrave/pytorch_metric_learning) and [record_keeper](https://github.com/KevinMusgrave/record_keeper)

In addition to tensorboard, all experiment data is automatically saved in pickle and CSV format in the ```saved_pkls``` subfolder.

## Override config options at the command line
The default config files use a batch size of 128. What if you want to use a batch size of 256? Just write the flag at the command line:
```
python run.py --experiment_name test2 --batch_size 256
```
All options in the config files can be overriden via the command line. This includes nested config options. For example, the default setting for ```mining_funcs``` (located in ```config/config_loss_and_miners/default.yaml```) is:
```yaml
mining_funcs:
  post_gradient_miner: 
    MultiSimilarityMiner: 
      epsilon: 0.1
```
If you want to use PairMarginMiner instead, you can do:
```
python run.py \
--experiment_name test2 \
--mining_funcs~OVERRIDE~ {post_gradient_miner: PairMarginMiner: {pos_margin: 0.5, neg_margin: 0.5}}}
```
Or if you don't want to use a miner at all:
```
python run.py \
--experiment_name test2 \
--mining_funcs~OVERRIDE~ {}
```
The ```~OVERRIDE~``` suffix is required to completely override complex config options. The reason is that by default, complex options are merged. For example, the default optimizers are:
```
optimizers:
  trunk_optimizer:
    Adam:
      lr: 0.00001
      weight_decay: 0.00005
  embedder_optimizer:
    Adam:
      lr: 0.00001
      weight_decay: 0.00005
```
If you want to add an optimizer for your loss function's parameters, just exclude the ```~OVERRIDE~``` suffix.
```
python run.py \
--experiment_name test2 \
--optimizers {metric_loss_optimizer: {SGD: {lr: 0.01}}} 
```
Now the ```optimizers``` parameter contains 3 optimizers because the command line flag was merged with the flag in the yaml file. To see more details about this functionality, check out [easy_module_attribute_getter](https://github.com/KevinMusgrave/easy_module_attribute_getter).

## Combine yaml files at the command line
The config files are currently separated into 6 folders, for readability. Suppose you want to try Deep Adversarial Metric Learning. You can write a new yaml file in the config_general folder that contains the necessary parameters. But there is no need to rewrite generic parameters like pytorch_home and num_epochs_train. Instead, just tell the program to use both the default config file and your new config file:
```
python run.py --experiment_name test3 --config_general default daml
```
With this command, ```configs/config_general/default.yaml``` will be loaded first, and then ```configs/config_general/daml.yaml``` will be merged into it. 

It turns out that [pytorch_metric_learning](https://github.com/KevinMusgrave/pytorch_metric_learning) allows you to run deep adversarial metric learning with a classifier layer. So you can write another yaml file containing the classifier layer parameters and optimizer, and then specify it on the command line:
```
python run.py --experiment_name test4 --config_general default daml train_with_classifier
```

## Resume training
To resume training from the most recently saved model, you just need to specify ```--experiment_name``` and ```--resume_training```.
```
python run.py --experiment_name test4 --resume_training
```
Let's say you finished training for 100 epochs, and decide you want to train for another 50 epochs, for a total of 150. You would run:
```
python run.py --experiment_name test4 --resume_training --num_epochs_train 150
```
Now in your experiments folder you'll see the original config files, and a new folder starting with ```resume_training```.
```
<root_experiment_folder>
|-<experiment_name>
  |-configs
    |-config_eval.yaml
    ...
    |-resume_training_config_diffs_1
  ...
```
This folder contains all differences between the originally saved config files and the parameters that you've specified at the command line. In this particular case, there should just be a single file ```config_general.yaml``` with a single line: ```num_epochs_train: 150```. Every time you resume training, a new folder will be created, showing what parameters changed since the original run.

## Evaluation options
By default, your model will be saved and evaluated on the training and validation sets every ```save_interval``` epochs. If you'd like to save the models but skip evaluation during training, set ```skip_eval``` to True. 

To get accuracy for specific splits, use the ```--splits_to_eval``` flag and pass in a space-delimited list of split names. For example ```--splits_to_eval train test```

To run evaluation only, use the ```--run_eval_only``` flag and pass in one of the following options:
- best (gets accuracy for the model that performed best on the validation set. This assumes you already calculated validation set accuracy)
- all (get accuracy for all saved models)
- a space-delimited list of integers, which are the model numbers you want to get accuracy for.

## Split schemes
One weakness of many metric-learning papers is that they have been training and testing on the same handful of datasets for years. They have also been splitting data into a 50/50 train/test split scheme, instead of train/val/test. This has likely lead to overfitting on the "test" set, as people have tuned hyperparameters and created algorithms with direct feedback from the "test" set.

To remedy this situation, this benchmarker allows the user to specify the split scheme via the ```split``` option. Currently there are 4 options. Each one splits data based on class labels. For example, the ```old_approach``` uses the first 50% of class labels for the training set, and the other 50% of labels for the "test" set (which should really be called the validation set).

| split scheme name | class split percentages | split names |
|-------------------|-------------------------|-------------|
| old_approach | 50/50 | train/val |
| hard | 20/5/75 | train/val/test |
| medium | 40/10/50 | train/val/test |
| easy | 60/15/25 | train/val/test |

To further increase the validity of accuracy measurements, you can use the ```num_variants_per_split_scheme``` option to specify the number of ways the dataset will be split. For example, if you use the ```old_approach``` with ```num_variants_per_split_scheme: 2```, then the experiment will train a model on the first 50% of classes, and then train another model on the second 50% of classes. Each experiment's data will be kept in a separate subfolder.   

## Config options guide
Below is the format for the various config files. Click on the links to see the default yaml file for each category.

### [config_general](https://github.com/KevinMusgrave/powerful_benchmarker/blob/master/configs/config_general/default.yaml)
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
check_untrained_accuracy: <boolean>

```
### [config_models](https://github.com/KevinMusgrave/powerful_benchmarker/blob/master/configs/config_models/default.yaml)
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
### [config_loss_and_miners](https://github.com/KevinMusgrave/powerful_benchmarker/blob/master/configs/config_loss_and_miners/default.yaml)
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
### [config_optimizers](https://github.com/KevinMusgrave/powerful_benchmarker/blob/master/configs/config_optimizers/default.yaml)
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
### [config_transforms](https://github.com/KevinMusgrave/powerful_benchmarker/blob/master/configs/config_transforms/default.yaml)
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
### [config_eval](https://github.com/KevinMusgrave/powerful_benchmarker/blob/master/configs/config_eval/default.yaml)
```yaml
eval_reference_set: <name> #options: compared_to_self, compared_to_sets_combined, compared_to_training_set
eval_normalize_embeddings: <boolean>
eval_use_trunk_output: <boolean>
eval_batch_size: <number>
eval_metric_for_best_epoch: <name> #options: NMI, precision_at_1, r_precision, mean_average_r_precision
eval_dataloader_num_workers: <number>
eval_pca: <number> or null #options: number of dimensions to reduce embeddings to via PCA, or null if you don't want to use PCA.
eval_size_of_tsne: <number> #The number of samples per split that you want to visualize via TSNE. Set to 0 if you don't want a TSNE plot.
```

## Acknowledgements
Thank you to Ser-Nam Lim at Facebook AI, and my research advisor, Professor Serge Belongie. This project began during my internship at Facebook AI where I received valuable feedback from Ser-Nam, and his team of computer vision and machine learning engineers and research scientists.
