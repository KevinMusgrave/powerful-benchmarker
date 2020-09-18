# Split Managers

Split managers take a dataset and form cross-validation splits based on some criteria.

## BaseSplitManager
Split managers must extend this class.

```python
from powerful_benchmarker.split_managers import BaseSplitManager
BaseSplitManager(hierarchy_level=0,
	        data_and_label_getter_keys=None,
	        labels_attr_name="labels", 
	        label_set_attr_name=None)
```

### Parameters

- ```hierarchy_level```: For multi-label datasets, this will select the set of labels that correspond to a specific column in the two-dimensional labels. For example, if the dataset has 1000 elements, and each element has 5 labels, then the label set has shape (1000, 5). If ```hierarchy_level=3```, then ```labels[:,3]``` will be used internally, if necessary.
- ```data_and_label_getter_keys```: If None, then calling ```dataset[idx]``` will simply return the raw value of ```dataset[idx]```. Otherwise, it will be assumed that ```dataset[idx]``` returns a dictionary, in which case ```data_and_label_getter_keys``` should be a list of strings that correspond to the dictionary keys for the data and labels.
- ```labels_attr_name```: The name of the dataset's attribute that contains the label for each element.
- ```label_set_attr_name```: The name of the dataset's attribute that contains the set of labels. If None, then the set will be computed using the dataset's list of labels.


## ClassDisjointSplitManager
Creates split schemes such that the trainval/test split has no overlap in class labels, and that each train/val split has no overlap in class labels. Extends [IndexSplitManager](split_managers.md#indexsplitmanager).
```python
from powerful_benchmarker.split_managers import ClassDisjointSplitManager
ClassDisjointSplitManager(**kwargs)
```


## ClosedSetSplitManager
Creates split schemes such that the ratios of class labels in the dataset is reflected in the trainval/test split, and in each train/val split. In other words, if the dataset has two classes, A and B, and 75% of the dataset is class A, then every train/val/test split will consist of roughly 75% class A. Extends [IndexSplitManager](split_managers.md#indexsplitmanager).
```python
from powerful_benchmarker.split_managers import ClosedSetSplitManager
ClosedSetSplitManager(**kwargs)
```

## IndexSplitManager

Creates split schemes based on dataset index. The logic within this class can be adapted for other cases, like splitting based on class label.

```python
from powerful_benchmarker.split_managers import IndexSplitManager
IndexSplitManager(num_training_partitions,
		        num_training_sets,
		        test_size=None,
		        test_start_idx=None,
		        shuffle = False,
		        random_seed = None,
		        helper_split_manager = None,
		        **kwargs)
```

### Parameters
- ```num_training_partitions```: The number of partitions in the trainval set. For example, if 40% of the dataset is used for the test set, and ```num_training_partitions = 2```, then 2 partitions of size (60%/2) = 30% will be created in the trainval set.
- ```num_training_sets```: The number of training and validation sets to create. Each validation set will have a size of one partition. For example, if ```num_training_partitions = 10```, then the validation set will always be one of those partitions, while the training set will comprise the other 9.
- ```test_size```: The size of the test size, as a floating point number. For example, ```test_size=0.4``` will make the test set 40% of the dataset.
- ```test_start_idx```: The location in the dataset where the test set starts. For example, if ```test_size=0.4``` and ```test_start_idx=0.5```, then the test set will consist of all dataset elements with indices between ```len(dataset)*0.5``` and ```len(dataset)*0.9```. It will wrap around to the beginning of the dataset if necessary.
- ```shuffle```: Whether or not to shuffle the dataset when forming the splits.
- ```random_seed```: A random seed for shuffling. Only applicable if ```shuffle = True```.
- ```helper_split_manager```: An optional external split manager. If provided, this external split manager will be used to create the trainval/test split. After that, IndexSplitManager will form the train/val splits.


## MLRCSplitManager

The split manager used for [A Metric Learning Reality Check](https://arxiv.org/abs/2003.08505). It is basically the same as [ClassDisjointSplitManager](split_managers.md#classdisjointsplitmanager), but with some differences in rounding and cross-validation fold order. Extends [IndexSplitManager](split_managers.md#indexsplitmanager).

```python
from powerful_benchmarker.split_managers import MLRCSplitManager
MLRCSplitManager(**kwargs)
```