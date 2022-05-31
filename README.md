<h1 align="center">
 Powerful Benchmarker
</h1>

## Benchmarking Validation Methods for Unsupervised Domain Adaptation

### Installation

Clone this repo:
```
git clone https://github.com/KevinMusgrave/powerful-benchmarker.git
```

Then go into the folder and install the required packages:
```
cd powerful-benchmarker
pip install -r requirements.txt
```

### Set paths in `constants.yaml`

- `exp_folder`: experiments will be saved as sub-folders inside of `exp_folder`
- `dataset_folder`: datasets will be downloaded here. For example, `<dataset_folder>/mnistm`
- `conda_env`: (optional) the conda environment that will be activated for slurm jobs
- `slurm_folder`: (optional) slurm logs will be saved to `<exp_folder>/.../<slurm_folder>`
- `gdrive_folder`: (optional) the google drive folder to which logs can be uploaded


### Folder organization
| Folder | Description |
| - | - |
| `latex` | Code for creating latex tables from experiment data.
| `notebooks` | Jupyter notebooks
| `powerful_benchmarker` | Code for hyperparameter searches for training models.
| `scripts` | Various bash scripts, including scripts for uploading logs to google drive.
| `unit_tests` | Tests to check if there are bugs.
| `validator_tests` | Code for evaluating validation methods (validators).



### Citing the paper

Coming soon...


## Looking for [A Metric Learning Reality Check](https://arxiv.org/pdf/2003.08505.pdf)?
Checkout the [metric-learning branch](https://github.com/KevinMusgrave/powerful-benchmarker/tree/metric-learning).
