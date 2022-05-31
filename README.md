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
- `dataset_folder`: datasets will be downloaded here. For example, `<dataset_folder>/mnistm`.
- `conda_env`: (optional) the conda environment that will be activated for slurm jobs
- `slurm_folder`: (optional) slurm logs will be saved to `<exp_folder>/.../<slurm_folder>`
- `gdrive_folder`: (optional) the google drive folder to which logs can be uploaded

### Getting started

#### Training a source-only model

Train a model on MNIST for 20 epochs:
```
python powerful_benchmarker/main.py --exp_name test_experiment0 --dataset mnist \
--src_domains mnist --adapter PretrainerConfig \
--download_datasets --num_trials 1 \
--max_epochs 20 --pretrain_on_src --validator src_accuracy \
--use_stat_getter
```

Test the source-only model on MNIST and MNISTM:
```
python powerful_benchmarker/main.py --exp_name test_experiment0 \
--target_domains mnist mnistm --evaluate --validator oracle
```

#### Hyperparameter search for domain adaptation algorithms

Train a model using DANN for 20 epochs, with 10 different random hyperparameter settings. Save features every 5 epochs.
```
python powerful_benchmarker/main.py --exp_name dann_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter DANNConfig \
--start_with_pretrained --feature_layer 6 \
--max_epochs 20 --num_trials 10 --save_features \
--val_interval 5
```

### Notebooks

The [notebooks](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/notebooks) folder currently contains:

- [ValidationScores.ipynb](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/notebooks/ValidationScores.ipynb)

### Citing the paper

Coming soon...


## Looking for [A Metric Learning Reality Check](https://arxiv.org/pdf/2003.08505.pdf)?
Checkout the [metric-learning branch](https://github.com/KevinMusgrave/powerful-benchmarker/tree/metric-learning).
