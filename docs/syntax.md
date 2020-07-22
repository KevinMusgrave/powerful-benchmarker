# Command Line Syntax



### Override config options at the command line
The default config files use a [batch size of 32](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/src/powerful_benchmarker/configs/config_models/default.yaml). You can override this default value at the command line:
```
python run.py --experiment_name test1 --batch_size 256
```

Complex options (i.e. nested dictionaries) can also be specified at the command line:
```
python run.py \
--experiment_name test1 \
--mining_funcs {tuple_miner: {PairMarginMiner: {pos_margin: 0.5, neg_margin: 0.5}}}
```
The ```~OVERRIDE~``` suffix is required to completely override complex config options. For example, the following overrides the [default loss function](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/src/powerful_benchmarker/configs/config_loss_and_miners/default.yaml):
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
This will be included along with the [default optimizers](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/src/powerful_benchmarker/configs/config_optimizers/default.yaml). 

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
python run.py --experiment_name test1 --config_general [default, with_cars196]
```
This is convenient when you want to change a few settings (specified in ```with_cars196```), and keep all the other options unchanged (specified in ```default```). You can specify any number of config files to merge, and they get loaded and merged in the order that you specify.
