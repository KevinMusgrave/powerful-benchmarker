# Datasets

When using powerful-benchmarker: 

  - ```root``` is set to ```dataset_root``` by default
  - ```transform``` is set based on the contents of ```config_transforms```, specifically the ```transforms``` config option. 

## Cars196
```python
from powerful_benchmarker.datasets import Cars196
Cars196(root, transform=None, download=False)
```

**Parameters**:

* **root**: Where the dataset is located, or where it will be downloaded.
* **transform**: The image transform that will be applied.
* **download**: Set to True to download the dataset to ```root```.


## CUB200
```python
from powerful_benchmarker.datasets import CUB200
CUB200(root, transform=None, download=False)
```

**Parameters**:

* **root**: Where the dataset is located, or where it will be downloaded.
* **transform**: The image transform that will be applied.
* **download**: Set to True to download the dataset to ```root```.

## StanfordOnlineProducts
```python
from powerful_benchmarker.datasets import StanfordOnlineProducts
StanfordOnlineProducts(root, transform=None, download=False)
```

**Parameters**:

* **root**: Where the dataset is located, or where it will be downloaded.
* **transform**: The image transform that will be applied.
* **download**: Set to True to download the dataset to ```root```.