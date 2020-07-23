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

## A Metric Learning Reality Check
This library was used for [A Metric Learning Reality Check](https://arxiv.org/abs/2003.08505). See [the documentation](https://kevinmusgrave.github.io/powerful-benchmarker/papers/mlrc) for supplementary material.

## Benchmark results: 
- [4-fold cross validation, test on 2nd-half of classes](https://docs.google.com/spreadsheets/d/1brUBishNxmld-KLDAJewIc43A4EVZk3gY6yKe8OIKbY/edit?usp=sharing)

## Benefits of this library
1. Highly configurable: 
   - [Yaml files](https://kevinmusgrave.github.io/powerful-benchmarker/yaml_syntax/) for organized configuration
   - A [powerful command line syntax](https://kevinmusgrave.github.io/powerful-benchmarker/cl_syntax/) that allows you to merge, override, swap, apply, and delete config options.
2. Customizable: 
   - Benchmark your own losses, miners, datasets etc. [with a simple function call](https://kevinmusgrave.github.io/powerful-benchmarker/custom/).
3. Easy hyperparameter optimization:
   - Simply [append the \~BAYESIAN\~ flag](https://kevinmusgrave.github.io/powerful-benchmarker/hyperparams/) to the names of hyperparameters you want to optimize.
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

## Citing the benchmark results
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

## Citing the code
If you'd like to cite the powerful-benchmarker code, you can use this bibtex:
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

## Acknowledgements
Thank you to Ser-Nam Lim at Facebook AI, and my research advisor, Professor Serge Belongie. This project began during my internship at Facebook AI where I received valuable feedback from Ser-Nam, and his team of computer vision and machine learning engineers and research scientists.
