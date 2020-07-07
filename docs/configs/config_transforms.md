# config_transforms

## transforms
Specifies the transforms to be used during training and during evaluation. ```ToTensor()``` and ```Normalize``` do not need to be specified, as they are added by default.


Default yaml:
```yaml
transforms:
  train:
    Resize:
      size: 256
    RandomResizedCrop:
      scale: 0.16 1
      ratio: 0.75 1.33
      size: 227
    RandomHorizontalFlip:
      p: 0.5

  eval:
    Resize:
      size: 256
    CenterCrop:
      size: 227
```

Command line:
```bash
--transforms {train: {Resize: {size: 256}, RandomResizedCrop: {scale: [0.16, 1], ratio: [0.75, 1.33], size: 227}, RandomHorizontalFlip: {p: 0.5}}, eval: {Resize: {size: 256}, CenterCrop: {size: 227}}}
```
