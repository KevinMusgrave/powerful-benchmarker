# Modules Available By Default 

With this library, objects are created based on the class names and parameters specified in the config files or at the command line. For example, ```shufflenet_v2_x1_0``` is one of the models in the ```torchvision.models``` module, so you can use it for your experiment like this:

In a config file:
```yaml
models:
  trunk:
    shufflenet_v2_x1_0:
      pretrained: True
```


At the command line:
```bash
--models~OVERRIDE~ {trunk: {shufflenet_v2_x1_0: {pretrained: True}}}
```

By default, the following modules are available.

  - Models
    - [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch){target=_blank}
    - [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html){target=_blank}
  - Optimizers
    - [torch.optim](https://pytorch.org/docs/stable/optim.html){target=_blank}
    - [torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate){target=_blank}
  - Datasets
    - [powerful_benchmarker.datasets](../code/datasets){target=_blank}
    - [torchvision.datasets](https://pytorch.org/docs/stable/torchvision/datasets.html){target=_blank}
  - Transforms
    - [torchvision.transforms.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html){target=_blank}
    - [easy_module_attribute_getter.custom_transforms](https://github.com/KevinMusgrave/easy-module-attribute-getter/blob/master/easy_module_attribute_getter/custom_transforms.py){target=_blank}
  - Losses
    - [pytorch_metric_learning.losses](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/){target=_blank}
    - [torch.nn](https://pytorch.org/docs/stable/nn.html){target=_blank}
  - Miners
    - [pytorch_metric_learning.miners](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/){target=_blank}
  - Samplers
    - [pytorch_metric_learning.samplers](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/){target=_blank}
    - [torch.utils.data](https://pytorch.org/docs/stable/data.html){target=_blank}
  - Trainers
    - [pytorch_metric_learning.trainers](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/){target=_blank}
  - Testers
    - [pytorch_metric_learning.testers](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/){target=_blank}


You can add other classes and modules by using the [register functionality](custom.md).