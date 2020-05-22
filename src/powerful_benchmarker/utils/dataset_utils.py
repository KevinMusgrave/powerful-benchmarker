#! /usr/bin/env python3

import torch.utils.data
import numpy as np
from . import common_functions as c_f

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