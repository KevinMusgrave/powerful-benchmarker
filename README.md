<h1 align="center">
 Powerful Benchmarker
</h1>

## [Unsupervised Domain Adaptation: A Reality Check](https://arxiv.org/pdf/2111.15672.pdf)

### Installation

Clone this repo, then:

```
pip install -r requirements.txt
```

### Set paths in ```constants.yaml```

- ```exp_folder```: experiments will be saved at ```<exp_folder>/<exp_name>```
- ```dataset_folder```: datasets will be downloaded here. For example, ```<dataset_folder>/mnistm``` and ```<dataset_folder>/office31```
- ```conda_env``` and ```slurm_folder``` are for running jobs on slurm. (I haven't uploaded the slurm-related code yet.)

### Running hyperparameter search

#### Example 1: DANN on MNIST->MNISTM task
```
python main.py --exp_name dann_experiment --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter DANNConfig \
--download_datasets --start_with_pretrained
```

#### Example 2: MCC on OfficeHome Art->Real task
```
python main.py --exp_name mcc_experiment --dataset officehome \
--src_domains art --target_domains real --adapter MCCConfig \
--download_datasets --start_with_pretrained
```

#### Example 3: Specify validator, batch size, etc.
```
python main.py --exp_name bnm_experiment --dataset office31 \
--src_domains dslr --target_domains amazon --adapter BNMConfig \
--batch_size 32 --max_epochs 500 --patience 15 \
--validation_interval 5 --num_workers 4 --num_trials 100 --n_startup_trials 100 \
--validator entropy_diversity --optimizer_name Adam \
--download_datasets --start_with_pretrained
```

### Note on algorithm/validator names
Some names in the code don't match the names in the paper. It would be good to change the names in the code, but I'm going to delay doing that, in case I have to rerun experiments and combine new dataframes with existing saved dataframes.

Here are the main differences between code and paper:

| Code | Paper |
| - | - |
| ```--validator entropy_diversity``` | Information Maximization (IM) validator |

### Notebooks

The [notebooks](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/notebooks) folder currently contains:

- [ValidationScores.ipynb](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/notebooks/ValidationScores.ipynb)

### Citing the paper

If you'd like to cite the paper, paste this into your latex bib file:
```
@misc{musgrave2021unsupervised,
      title={Unsupervised Domain Adaptation: A Reality Check}, 
      author={Kevin Musgrave and Serge Belongie and Ser-Nam Lim},
      year={2021},
      eprint={2111.15672},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Looking for [A Metric Learning Reality Check](https://arxiv.org/pdf/2003.08505.pdf)?
Checkout the [metric-learning branch](https://github.com/KevinMusgrave/powerful-benchmarker/tree/metric-learning).
