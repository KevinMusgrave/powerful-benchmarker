<h1 align="center">
 <a href="https://arxiv.org/abs/2003.08505">A Metric Learning Reality Check</a>
</h2>
<p align="center">


## [Benchmark results (in progress)](https://drive.google.com/open?id=1Y_stkiqlHA7HTMNrhyPCnYhR0oevphRR): 
- [Spreadsheet #1: Train/val 50/50](https://docs.google.com/spreadsheets/d/1kiJ5rKmneQvnYKpVO9vBFdMDNx-yLcXV2wbDXlb-SB8/edit?usp=sharing)
- [Spreadsheet #2: 4-fold cross validation, test on 2nd-half of classes](https://docs.google.com/spreadsheets/d/1brUBishNxmld-KLDAJewIc43A4EVZk3gY6yKe8OIKbY/edit?usp=sharing)

## Benefits of this library
1. Highly configurable
    - Use the default configs files, merge in your own, or override options via the command line.
2. Extensive logging
    - View experiment data in tensorboard, csv, and sqlite format.
3. Easy hyperparameter optimization
    - Simply append \~BAYESIAN\~ to the hyperparameters you want to optimize.
4. Customizable
    - Register your own losses, miners, datasets etc. with a simple function call.

## Installation
```
pip install powerful-benchmarker
pip install pytorch-metric-learning==0.9.82.dev0
```

## Usage

### Set default flags

The easiest way to get started is to download the [example script](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/examples/run.py). Then change the default values for the following flags:

- pytorch_home is where you want to save downloaded pretrained models.
- dataset_root is where your datasets are located.
- root_experiment_folder is where you want all experiment data to be saved.


### Download and organize the datasets
Download the datasets from here:
- [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [Cars196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [Stanford Online Products](http://cvgl.stanford.edu/projects/lifted_struct)

Organize them as follows:
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

### Try a basic command
The following command will run an experiment using the [default config files](https://github.com/KevinMusgrave/powerful-benchmarker/tree/master/powerful_benchmarker/configs)
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
    |-saved_csvs
    |-tensorboard_logs
  |-meta_logs
    |-saved_csvs
    |-tensorboard_logs
```

### Override config options at the command line
The default config files use a [batch size of 32](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/powerful_benchmarker/configs/config_models/default.yaml). What if you want to use a batch size of 256? Just write the flag at the command line:
```
python run.py --experiment_name test1 --batch_size 256
```
Complex options (i.e. nested dictionaries) can be specified at the command line:
```
python run.py \
--experiment_name test1 \
--mining_funcs {tuple_miner: {PairMarginMiner: {pos_margin: 0.5, neg_margin: 0.5}}}
```
The ```~OVERRIDE~``` suffix is required to completely override complex config options. For example, the following overrides the [default loss function](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/powerful_benchmarker/configs/config_loss_and_miners/default.yaml):
```
python run.py \
--experiment_name test1 \
--loss_funcs {metric_loss~OVERRIDE~: {ArcFaceLoss: {margin: 30, scale: 64, embedding_size: 128}}}
```
Leave out the ```~OVERRIDE~``` suffix if you want to merge options. For example, we can add an optimizer for our loss function's parameters:
```
python run.py \
--experiment_name test1 \
--optimizers {metric_loss_optimizer: {SGD: {lr: 0.01}}} 
```
This will be included along with the [default optimizers](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/powerful_benchmarker/configs/config_optimizers/default.yaml). 

We can change the learning rate of the trunk_optimizer, but keep all other parameters the same:
```
python run.py \
--experiment_name test1 \
--optimizers {trunk_optimizer: {RMSprop: {lr: 0.01}}} 
```

Or we can make trunk_optimizer use Adam, but leave embedder_optimizer to the default setting: 
```
python run.py \
--experiment_name test1 \
--optimizers {trunk_optimizer~OVERRIDE~: {Adam: {lr: 0.01}}} 
```

### Combine yaml files at the command line
The following merges the ```with_cars196``` config file into the ```default``` config file, in the ```config_general``` category.
```
python run.py --experiment_name test1 --config_general default with_cars196
```
This is convenient when you want to change a few settings (specified in ```with_cars196```), and keep all the other options unchanged (specified in ```default```). You can specify any number of config files to merge, and they get loaded and merged in the order that you specify.

### Resume training
The following resumes training for the ```test1``` experiment:
```
python run.py --experiment_name test1 --resume_training
```
Let's say you finished training for 100 epochs, and decide you want to train for another 50 epochs, for a total of 150. You would run:
```
python run.py --experiment_name test1 --resume_training --num_epochs_train 150
```
Now in your experiments folder you'll see the original config files, and a new folder starting with ```resume_training```.
```
<root_experiment_folder>
|-<experiment_name>
  |-configs
    |-config_eval.yaml
    ...
    |-resume_training_config_diffs_<underscore delimited numbers>
  ...
```
This folder contains all differences between the originally saved config files and the parameters that you've specified at the command line. In this particular case, there should just be a single file ```config_general.yaml``` with a single line: ```num_epochs_train: 150```. 

The underscore delimited numbers in the folder name indicate which models were loaded for each [split scheme](#split-schemes-and-cross-validation). For example, let's say you are doing cross validation with 3 folds. The training process has finished 50, 30, and 0 epochs of folds 0, 1, and 2, respectively. You decide to stop training, and resume training with a different batch size. Now the config diff folder will be named ```resume_training_config_diffs_50_30_0```.

### Reproducing benchmark results
To reproduce an experiment from the benchmark spreadsheets, use the ```--reproduce_results``` flag:
1. In the benchmark spreadsheet, click on the google drive link under the "config files" column.
2. Download the folders you want (for example ```cub200_old_approach_triplet_batch_all```), into some folder on your computer. For example, I downloaded into ```/home/tkm45/experiments_to_reproduce```
3. Then run:
```
python run.py --reproduce_results /home/tkm45/experiments_to_reproduce/cub200_old_approach_triplet_batch_all \
--experiment_name cub200_old_approach_triplet_batch_all_reproduced
```

### Evaluation options
By default, your model will be saved and evaluated on the validation set every ```save_interval``` epochs.

To get accuracy for specific splits, use the ```--splits_to_eval``` flag and pass in a space-delimited list of split names. For example ```--splits_to_eval train test```

To run evaluation only, use the ```--evaluate``` flag.

### Split schemes and cross validation
One weakness of many metric-learning papers is that they have been training and testing on the same handful of datasets for years. They have also been splitting data into a 50/50 train/test split scheme, instead of train/val/test. This has likely lead to overfitting on the "test" set, as people have tuned hyperparameters and created algorithms with direct feedback from the "test" set.

To remedy this situation, this benchmarker allows the user to specify the split scheme. Here's an example config:
```yaml
test_size: 0.5
test_start_idx: 0.5
num_training_partitions: 10
num_training_sets: 5
```
Translation:
- The test set consists of classes with labels in ```[num_labels * test_start_idx, num_labels * (test_start_idx + test_size)]```. Note that if we set ```test_start_idx``` to 0.9, the range would wrap around to the beginning (0.9 to 1, 0 to 0.4). 
- The remaining classes will be split into 10 equal sized partitions. 
- 5 of those partitions will be used for training. In other words, 5-fold cross validation will be performed, but the size of the partitions will be the same as if 10-fold cross validation was being performed.

When evaluating the cross-validated models, the best model from each fold will be loaded, and the results be averaged. Alternatively, you can set the config option ```meta_testing_method``` to ```ConcatenateEmbeddings```. This will load the best model from each fold, but treat them as one model during evaluation on the test set, by concatenating their outputs.

If instead you still want to use the old 50/50 train/test split, then set ```special_split_scheme_name``` to ```old_approach```. Otherwise, leave it as ```null```. 

### Meta logs
When doing cross validation, a new set of meta records will be created. The meta records show the average of the best accuracies of your training runs. You can find these records on tensorboard and in the meta_logs folder.

### Bayesian optimization to tune hyperparameters
You can use bayesian optimization using the same [example script](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/examples/run.py). In your config files or at the command line, append ```~BAYESIAN~``` to any parameter that you want to tune, followed by a lower and upper bound in square brackets. If your parameter operates on a log scale (for example, learning rates), then append ```~LOG_BAYESIAN~```. You must also specify the number of iterations with the ```--bayes_opt_iters``` command line flag.

Here is an example script which uses bayesian optimization to tune 3 hyperparameters for the multi similarity loss.
```
python run.py --bayes_opt_iters 50 \
--loss_funcs~OVERRIDE~ {metric_loss: {MultiSimilarityLoss: {alpha~LOG_BAYESIAN~: [0.01, 100], beta~LOG_BAYESIAN~: [0.01, 100], base~BAYESIAN~: [0, 1]}}} \
--experiment_name cub_bayes_opt \
```

If you stop and want to resume bayesian optimization, simply use ```run.py``` with the same ```experiment_name``` you were using before. (Do not use the ```resume_training``` flag.) 

You can also run a number of reproductions for the best parameters, so that you can obtain a confidence interval for your results. Use the ```reproductions``` flag, and pass in the number of reproductions you want to perform at the end of bayesian optimization.

```
python run.py --bayes_opt_iters 50 --reproductions 10 \
--experiment_name cub_bayes_opt \
```

### Register your own classes and modules
By default, the API gives you access to losses/miners/datasets/optimizers/schedulers/trainers etc that are available in powerful-benchmarker, PyTorch, and pytorch-metric-learning.

Let's say you make your own loss and mining functions, and you'd like to have access to them via the API. You can accomplish this by replacing the last two lines of the [example script](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/examples/run.py) with this:

```python
from pytorch_metric_learning import losses, miners

# your custom loss function
class YourLossFunction(losses.BaseMetricLossFunction):
   ...

# your custom mining function
class YourMiningFunction(miners.BaseTupleMiner):
   ...

r = runner(**(args.__dict__))

# make the runner aware of them
r.register("loss", YourLossFunction)
r.register("miner", YourMiningFunction)
r.run()
```

Now you can access your custom classes just like any other class:
```yaml
loss_funcs:
  metric_loss: 
    YourLossFunction:

mining_funcs:
  tuple_miner:
    YourMiningFunction:
```

If you have a module containing multiple classes and you want to register all those classes, you can simply register the module:
```python
import YourModuleOfLosses

r.register("loss", YourModuleOfLosses)
```


## Config options guide
Below is the format for the various config files. Click on the links to see the default yaml file for each category.

### [config_general](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/configs/config_general/default.yaml)
```yaml
training_method: <type> #options: MetricLossOnly, TrainWithClassifier, CascadedEmbeddings, DeepAdversarialMetricLearning
testing_method: <type> #options: GlobalEmbeddingSpaceTester, WithSameParentLabelTester
meta_testing_method: <type> #options: null or ConcatenateEmbeddings
dataset:  
  <type>: #options: CUB200, Cars196, StanfordOnlineProducts
    <kwarg>: <value>
    ...
num_epochs_train: <how long to train for>
iterations_per_epoch: <how long an "epoch" lasts>
save_interval: <how often (in number of epochs) models will be saved and evaluated>
special_split_scheme_name: <string> #options: old_approach or predefined. Leave as null if you want to do cross validation.
test_size: <number> #number in (0, 1), which is the percent of classes that will be used in the test set.
test_start_idx: <number> #number in (0, 1), which is the percent that specifies the starting class index for the test set
num_training_partitions: <int> #number of partitions (excluding the test set) that are created for cross validation.
num_training_sets: <int> #number of partitions that are actually used as training sets cross validation.

label_hierarchy_level: <number>
dataloader_num_workers: <number>
check_untrained_accuracy: <boolean>
patience: <int> #Training will stop if validation accuracy has not improved after this number of epochs. If null, then it is ignored.

```
### [config_models](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/configs/config_models/default.yaml)
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
### [config_loss_and_miners](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/configs/config_loss_and_miners/default.yaml)
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
### [config_optimizers](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/configs/config_optimizers/default.yaml)
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
### [config_transforms](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/configs/config_transforms/default.yaml)
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
### [config_eval](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/configs/config_eval/default.yaml)
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

## Citing this library
If you'd like to cite powerful-benchmarker in your paper, you can use this bibtex:
```latex
@misc{Musgrave2019,
  author = {Musgrave, Kevin and Lim, Ser-Nam and Belongie, Serge},
  title = {Powerful Benchmarker},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/KevinMusgrave/powerful-benchmarker}},
}
```
