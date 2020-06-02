#! /usr/bin/env python3

import torch.utils.data
import numpy as np
from . import common_functions as c_f
from collections import OrderedDict

def get_underlying_dataset(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        return dataset.dataset
    return dataset

def get_dataset_attr(dataset, attr_name):
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = get_underlying_dataset(dataset)
    return c_f.get_attr_and_try_as_function(dataset, attr_name)    

def get_dataset_labels(dataset, labels_attr_name):
    labels = np.array(get_dataset_attr(dataset, labels_attr_name))
    if isinstance(dataset, torch.utils.data.Subset):
        return labels[dataset.indices]
    return labels

def get_labels_by_hierarchy(labels, hierarchy_level):
    if labels.ndim == 2:
        labels = labels[:, hierarchy_level]
    return labels

def get_label_set(labels, hierarchy_level):
    L = get_labels_by_hierarchy(np.array(labels), hierarchy_level)
    return set(L)

def create_subset(dataset, idx_to_keep):
    assert max(idx_to_keep) < len(dataset)
    return torch.utils.data.Subset(dataset, idx_to_keep)


def create_subset_datasets_from_indices(datasets, 
                                    trainval_subset_idx,
                                    trainval_global_idx, 
                                    test_set_idx, 
                                    split_scheme_name_func, 
                                    num_training_sets,
                                    create_subset_idx_func):
    split_schemes = OrderedDict()
    for i, (train_idx, val_idx) in enumerate(trainval_subset_idx):
        if i < num_training_sets:
            split_dict = {"train": trainval_global_idx[train_idx], "val": trainval_global_idx[val_idx], "test": test_set_idx}
            name = split_scheme_name_func(i)
            split_schemes[name] = OrderedDict()
            for transform_type in datasets.keys():
                split_schemes[name][transform_type] = OrderedDict()
                for k, v in split_dict.items():
                    curr_dataset = datasets[transform_type][k]
                    subset_idx = create_subset_idx_func(curr_dataset, v)
                    split_schemes[name][transform_type][k] = create_subset(curr_dataset, subset_idx)
    return split_schemes