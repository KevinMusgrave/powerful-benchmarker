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
