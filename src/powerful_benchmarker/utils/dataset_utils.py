#! /usr/bin/env python3

import torch.utils.data
import numpy as np

def get_subset_dataset_labels(subset_dataset):
    return subset_dataset.dataset.labels[subset_dataset.indices]

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