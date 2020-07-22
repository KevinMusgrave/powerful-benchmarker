<h1 align="center">
 <a href="https://arxiv.org/abs/2003.08505">A Metric Learning Reality Check</a>
</h2>
<p align="center">
	
</h2>
<p align="center">
 <a href="https://badge.fury.io/py/powerful-benchmarker">
     <img alt="PyPi version" src="https://badge.fury.io/py/powerful-benchmarker.svg">
 </a>
 

## Documentation
[**View the documentation here**](https://kevinmusgrave.github.io/powerful-benchmarker/)

## Benchmark results: 
- [4-fold cross validation, test on 2nd-half of classes](https://docs.google.com/spreadsheets/d/1brUBishNxmld-KLDAJewIc43A4EVZk3gY6yKe8OIKbY/edit?usp=sharing)

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
