#! /usr/bin/env python3

from collections import OrderedDict, defaultdict

import numpy as np
import torch.utils.data


def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    labels_to_indices = {}
    if labels.ndim == 1:
        labels_to_indices[0] = defaultdict(list)
        for i, label in enumerate(labels):
            labels_to_indices[0][label].append(i)
    else:
        for i in range(labels.shape[1]):
            labels_to_indices[i] = defaultdict(list)
        for i, label_list in enumerate(labels):
            for j, label in enumerate(label_list):
                labels_to_indices[j][label].append(i)

    for _, v1 in labels_to_indices.items():
        for k2, v2 in v1.items():
            v1[k2] = np.array(v2, dtype=np.int)

    return labels_to_indices


def get_label_based_split_scheme_ratios(split_scheme_name):
    """
    Args:
        split_scheme_name: type string, the name of the split scheme to be returned
    Returns:
        A dict which maps train/val/test to percentages of the dataset classes
        that will be used for a particular split
    In most metric learning papers, the datasets are split into train and test,
    where train and test take the first and second half of classes respectively.
    To better evaluate algorithms, we want to try different split schemes.
    For example the hard split trains on 20% of classes, validates on 5% of
    classes, and tests on the remaining 75% of classes.
    """
    d = {
        "old_approach": {"train": 0.5, "val": 0.5},
        "hard": {"train": 0.2, "val": 0.05, "test": 0.75},
        "medium": {"train": 0.4, "val": 0.1, "test": 0.5},
        "easy": {"train": 0.6, "val": 0.15, "test": 0.25},
    }
    return d[split_scheme_name]


def create_subset(dataset, idx_to_keep):
    """
    Args:
        dataset: type torch.utils.data.Dataset, the dataset to return a subset of
        labels: type torch.tensor, the labels for every element in the dataset
        idx_to_keep: type sequence, the indices of the elements that will form
                     the subset
    Returns:
        subset_dataset: type torch.utils.data.Dataset
        subset_labels: type torch.tensor
    """
    assert max(idx_to_keep) < len(dataset)
    subset_dataset = torch.utils.data.Subset(dataset, idx_to_keep)
    return subset_dataset


def create_label_based_subset(dataset, labels, class_rule):
    """
    This behaves like create_subset, except for the third argument.
    class_rule is a function that takes in a label (an integer) and outputs
    True if that class should be included in the subset dataset.
    """
    idx_to_keep = [i for i, label in enumerate(labels) if class_rule(label)]
    return create_subset(dataset, idx_to_keep), idx_to_keep


def numeric_class_rule(rule_params, mode="range"):
    """
    Args:
        rule_params: a list, the purpose of which is determined by the mode specified
    Returns:
        a function that takes a label as input, and returns True if that label
                    should be included in a subset dataset.
    Currently there is only the "range" mode, but that might change later.
    """
    ### maybe add other modes later
    a, b = rule_params
    if a < b:
        return lambda label: a <= label <= b
    elif a > b:
        return lambda label: label >= a or label <= b


def create_one_split_scheme(dataset, split_scheme_name, start_idx=0, hierarchy_level=0):
    """
    Args:
        dataset: type torch.utils.data.Dataset, the dataset to return a subset of
        split_scheme_name: type string, the name of the split scheme to be returned
    Returns:
        A dictionary where each key is a split name (i.e. "train", "val", "test"),
                and the value is a tuple: (subset_dataset, subset_labels)
    """
    traintest_dict = OrderedDict()
    if split_scheme_name == "predefined":
        for k, v in dataset.predefined_splits.items():
            traintest_dict[k] = torch.utils.data.Subset(dataset, v), v
    else:
        labels = dataset.labels
        if labels.ndim == 2:
            labels = labels[:, hierarchy_level]
        sorted_label_set = sorted(list(set(labels)))
        num_original_labels = len(sorted_label_set)
        class_ratios = get_label_based_split_scheme_ratios(split_scheme_name)
        split_lens = {k: int(num_original_labels * v) for k, v in class_ratios.items()}
        split_lens["train"] += num_original_labels - sum(
            v for k, v in split_lens.items()
        )
        assert sum(v for _, v in split_lens.items()) == num_original_labels

        start_idx = int(start_idx*len(sorted_label_set))
        for k, v in split_lens.items():
            s = sorted_label_set[start_idx % len(sorted_label_set)]
            e = sorted_label_set[(start_idx + v - 1) % len(sorted_label_set)]
            class_rule = numeric_class_rule([s, e], mode="range")
            traintest_dict[k] = create_label_based_subset(dataset, labels, class_rule)
            start_idx += v
        # add other split schemes if we want

    return traintest_dict


def make_label_to_rank_dict(label_set):
    """
    Args:
        label_set: type sequence, a set of integer labels
                    (no duplicates in the sequence)
    Returns:
        A dictionary mapping each label to its numeric rank in the original set
    """
    argsorted = list(np.argsort(label_set))
    return {k: v for k, v in zip(label_set, argsorted)}
