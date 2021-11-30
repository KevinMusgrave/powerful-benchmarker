<h1 align="center">
 Powerful Benchmarker
</h2>
<p align="center">
	
</h2>
<p align="center">
 <a href="https://badge.fury.io/py/powerful-benchmarker">
     <img alt="PyPi version" src="https://badge.fury.io/py/powerful-benchmarker.svg">
 </a>
 

## Documentation
[**View the documentation here**](https://kevinmusgrave.github.io/powerful-benchmarker/)

## Google Colab Examples
See the [examples folder](https://github.com/KevinMusgrave/powerful-benchmarker/tree/master/examples) for notebooks that show a bit of this library's functionality.

## A Metric Learning Reality Check
See [**supplementary material**](https://kevinmusgrave.github.io/powerful-benchmarker/papers/mlrc) for the [ECCV 2020 paper](https://arxiv.org/abs/2003.08505).

## Benchmark results: 
- [4-fold cross validation, test on 2nd-half of classes](https://docs.google.com/spreadsheets/d/1brUBishNxmld-KLDAJewIc43A4EVZk3gY6yKe8OIKbY/edit?usp=sharing)

## Benefits of this library
1. Highly configurable: 
   - [Yaml files](https://kevinmusgrave.github.io/powerful-benchmarker/yaml_syntax/) for organized configuration
   - A [powerful command line syntax](https://kevinmusgrave.github.io/powerful-benchmarker/cl_syntax/) that allows you to merge, override, swap, apply, and delete config options.
2. Customizable: 
   - Benchmark your own losses, miners, datasets etc. [with a simple function call](https://kevinmusgrave.github.io/powerful-benchmarker/custom/).
3. Easy hyperparameter optimization:
   - [Append the \~BAYESIAN\~ flag](https://kevinmusgrave.github.io/powerful-benchmarker/hyperparams/) to the names of hyperparameters you want to optimize.
4. Extensive logging:
   - View experiment data in [tensorboard, CSV and SQLite format](https://kevinmusgrave.github.io/powerful-benchmarker/#view-experiment-data).
5. Reproducible:
   - Config files are saved with each experiment and are [easily reproduced](https://kevinmusgrave.github.io/powerful-benchmarker/#reproduce-an-experiment).
6. Trackable changes:
   - [Keep track of changes](https://kevinmusgrave.github.io/powerful-benchmarker/#keep-track-of-changes) to an experiment's configuration.

## Installation
```
pip install powerful-benchmarker
```

## Citing the benchmark results or code
If you'd like to cite the benchmark results, please cite this paper:
```latex
@misc{musgrave2020metric,
    title={A Metric Learning Reality Check},
    author={Kevin Musgrave and Serge Belongie and Ser-Nam Lim},
    year={2020},
    eprint={2003.08505},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgements
Thank you to Ser-Nam Lim at Facebook AI, and my research advisor, Professor Serge Belongie. This project began during my internship at Facebook AI where I received valuable feedback from Ser-Nam, and his team of computer vision and machine learning engineers and research scientists.
