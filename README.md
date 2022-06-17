<h1 align="center">
 Powerful Benchmarker
</h1>

## Benchmarking Validation Methods for Unsupervised Domain Adaptation

## Installation

Clone this repo:
```
git clone https://github.com/KevinMusgrave/powerful-benchmarker.git
```

Then go into the folder and install the required packages:
```
cd powerful-benchmarker
pip install -r requirements.txt
```

## Set paths in `constants.yaml`

- `exp_folder`: experiments will be saved as sub-folders inside of `exp_folder`
- `dataset_folder`: datasets will be downloaded here. For example, `<dataset_folder>/mnistm`
- `conda_env`: (optional) the conda environment that will be activated for slurm jobs
- `slurm_folder`: slurm logs will be saved to `<exp_folder>/.../<slurm_folder>`
- `gdrive_folder`: (optional) the google drive folder to which logs can be uploaded


## Folder organization

Visit each folder to view its readme file.

| Folder | Description |
| - | - |
| [`latex`](https://github.com/KevinMusgrave/powerful-benchmarker/tree/master/latex) | Code for creating latex tables from experiment data.
| [`notebooks`](https://github.com/KevinMusgrave/powerful-benchmarker/tree/master/notebooks) | Jupyter notebooks
| [`powerful_benchmarker`](https://github.com/KevinMusgrave/powerful-benchmarker/tree/master/powerful_benchmarker) | Code for hyperparameter searches for training models.
| [`scripts`](https://github.com/KevinMusgrave/powerful-benchmarker/tree/master/scripts) | Various bash scripts, including scripts for uploading logs to google drive.
| [`unit_tests`](https://github.com/KevinMusgrave/powerful-benchmarker/tree/master/unit_tests) | Tests to check if there are bugs.
| [`validator_tests`](https://github.com/KevinMusgrave/powerful-benchmarker/tree/master/validator_tests) | Code for evaluating validation methods (validators).


## Useful top-level scripts

### delete_slurm_logs.py
Delete all slurm logs:
```
python delete_slurm_logs.py --delete
```

Or delete slurm logs for specific experiments groups. For example, delete slurm logs for all experiment groups starting with "officehome":
```
python delete_slurm_logs.py --delete --exp_group_prefix officehome
```
---
### kill_all.py
Kill all model training jobs:
```
python kill_all.py
```
Or kill all validator test jobs:
```
python kill_all.py --validator_tests
```
---
### print_progress.py
Print how many hyperparameter trials are done:
```
python print_progress.py
```

Include a detailed summary of validator test jobs:
```
python print_progress.py --with_validator_progress
```

Save to `progress.txt` instead of printing to screen:
```
python print_progress.py --save_to_file progress.txt
```
---
### simple_slurm.py
A simple way to run a program via slurm. 

For example, run `collect_dfs.py` for all experiment groups starting with "office31", using a separate slurm job for each experiment group:
```
python simple_slurm.py --command "python validator_tests/collect_dfs.py" --slurm_config_folder validator_tests \
--slurm_config a100 --job_name=collect_dfs --cpus-per-task=16 --exp_group_prefix office31
```

Or run a program without considering experiment groups at all:
```
python simple_slurm.py --command "python validator_tests/zip_dfs.py" --slurm_config_folder validator_tests \
--slurm_config a100 --job_name=zip_dfs --cpus-per-task=16
```
---
### upload_logs.py
Upload slurm logs and experiment progress to a google drive folder at regular intervals (the default is every 2 hours):
```
python upload_logs.py
```
Set the google drive folder in `constants.yaml`.



## Citing the paper

Coming soon...


## Looking for [A Metric Learning Reality Check](https://arxiv.org/pdf/2003.08505.pdf)?
Checkout the [metric-learning branch](https://github.com/KevinMusgrave/powerful-benchmarker/tree/metric-learning).
