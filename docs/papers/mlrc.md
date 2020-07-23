# A Metric Learning Reality Check

This page contains additional information for the [ECCV 2020 paper](https://arxiv.org/abs/2003.08505) by Musgrave et al.

## Reproducing results
### Download the experiment folder

1. Go to the [benchmark spreadsheet](https://docs.google.com/spreadsheets/d/1brUBishNxmld-KLDAJewIc43A4EVZk3gY6yKe8OIKbY/)
2. Find the experiment you want to reproduce, click on the link in the "Config files" column.
3. You'll see 3 folders: one for CUB, one for Cars, and one for SOP. Open the folder for the dataset you want to train on.
4. Now you'll see several files and folders, one of which ends in "reproduction0". Download this folder. (This folder will include saved models. If you don't want to download the saved models, go into the folder and download just the "configs" folder.)

### Command line scripts
Normally reproducing results is as easy as downloading an experiment folder, and [using the ```reproduce_results``` flag](../index.md#reproduce-an-experiment). However, there have been significant changes to the API since these experiments were run, so there are a couple of extra steps required, and they depend on the dataset:

  - CUB200:

```bash
python run.py --reproduce_results <experiment_to_reproduce> \
--experiment_name <your_experiment_name>
```

  - Cars196:

```bash
python run.py --reproduce_results <experiment_to_reproduce> \
--experiment_name <your_experiment_name> \
--config_dataset [default, with_cars196] \
--config_general [default, with_cars196] \
--merge_argparse_when_resuming
```

  - Stanford Online Products

```bash
python run.py --reproduce_results <experiment_to_reproduce> \
--experiment_name <your_experiment_name> \
--config_dataset [default, with_sop] \
--config_general [default, with_sop] \
--merge_argparse_when_resuming
```

  - CUB200 with batch size 256:
```bash
python run.py --reproduce_results <experiment_to_reproduce> \
--experiment_name <your_experiment_name> \
--trainer~APPLY~2: {batch_size: 256}
```

If you don't have the datasets and would like to download them into your ```dataset_root``` folder, you can add this flag to the CUB commands:
```bash
--dataset~OVERRIDE~ {CUB200: {download: True}}
```

Likewise, for the Cars196 and Stanford Online Products commands, replace the ```--config_dataset``` flag with:
```bash
--dataset~OVERRIDE~ {Cars196: {download: True}}
```
or 
```bash
--dataset~OVERRIDE~ {StanfordOnlineProducts: {download: True}}
```


## Frequently Asked Questions

#### Isn't it unfair to fix the model, optimizer, learning rate, and embedding size?

#### Why weren't more hard-mining methods evaluated?

#### Why was the batch size set to 32 for most of the results?

#### What is the difference between MAP@R and MAP?

#### For the contrastive loss, why is the optimal positive margin a negative value?

- A negative value should be equivalent to a margin of 0, because the distance between positive pairs cannot be negative, and the margin does not contribute to the gradient. So allowing the hyperparameter optimization to explore negative margins was unnecesary.


## Optimization plots


