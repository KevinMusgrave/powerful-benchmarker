#! /usr/bin/env python3

from collections import OrderedDict, defaultdict

import numpy as np
import torch.utils.data


def get_labels_by_hierarchy(labels, hierarchy_level):
    if labels.ndim == 2:
        labels = labels[:, hierarchy_level]
    return labels

def get_base_split_name(test_size, test_start_idx, num_training_partitions, partition=''):
    test_size = int(test_size*100)
    test_start_idx = int(test_start_idx*100)
    return 'Test%02d_%02d_Partitions%d_%s'%(test_size, test_start_idx, num_training_partitions, partition)


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
    return create_subset(dataset, idx_to_keep)


def numeric_class_rule(a, b, exclusion_rule=None):
    """
    Args:
        rule_params: a list, the purpose of which is determined by the mode specified
    Returns:
        a function that takes a label as input, and returns True if that label
                    should be included in a subset dataset.
    """
    exclusion_rule = (lambda label: True) if exclusion_rule is None else exclusion_rule

    if a < b:
        range_rule = lambda label: (a <= label <= b)
    elif a > b:
        range_rule = lambda label: (label >= a or label <= b)
    return lambda label: (range_rule(label) and exclusion_rule(label))


def get_wrapped_range(start_idx, length_of_range, length_of_list):
    s = start_idx % length_of_list
    e = (start_idx + length_of_range - 1) % length_of_list
    return s, e

def get_single_class_rule(start_idx, split_length, sorted_label_set, exclusion_rule=None):
    s, e = get_wrapped_range(start_idx, split_length, len(sorted_label_set))
    s = sorted_label_set[s]
    e = sorted_label_set[e]
    return numeric_class_rule(s, e, exclusion_rule)

def get_class_rules(start_idx, split_lengths, sorted_label_set, exclusion_rule=None):
    class_rules = {}
    for k, v in split_lengths.items():
        class_rules[k] = get_single_class_rule(start_idx, v, sorted_label_set, exclusion_rule)
        start_idx += v
    return class_rules

def split_lengths_from_ratios(class_ratios, num_labels):
    split_lengths = {k: int(num_labels * v) for k, v in class_ratios.items()}
    split_lengths["train"] += num_labels - sum(v for k, v in split_lengths.items())
    assert sum(v for _, v in split_lengths.items()) == num_labels
    return split_lengths

def create_one_split_scheme(dataset, scheme_name=None, partition=None, num_training_partitions=None, test_size=None, test_start_idx=None, hierarchy_level=0):
    """
    Args:
        dataset: type torch.utils.data.Dataset, the dataset to return a subset of
        split_scheme_name: type string, the name of the split scheme to be returned
    Returns:
        A dictionary where each key is a split name (i.e. "train", "val", "test"),
                and the value is a tuple: (subset_dataset, subset_labels)
    """
    traintest_dict = OrderedDict()
    if scheme_name == "predefined":
        for k, v in dataset.predefined_splits.items():
            traintest_dict[k] = torch.utils.data.Subset(dataset, v)
    else:
        labels = get_labels_by_hierarchy(dataset.labels, hierarchy_level)
        sorted_label_set = sorted(list(set(labels)))
        num_labels = len(sorted_label_set)

        if scheme_name == "old_approach":
            split_lengths = split_lengths_from_ratios({"train": 0.5, "val": 0.5}, num_labels)
            for k, class_rule in get_class_rules(0, split_lengths, sorted_label_set).items():
                traintest_dict[k] = create_label_based_subset(dataset, labels, class_rule)
        else:
            val_ratio = (1./num_training_partitions)*(1-test_size)
            train_ratio = (1. - val_ratio)*(1-test_size)
            class_ratios = {"train": train_ratio, "val": val_ratio, "test": test_size}
            split_lengths = split_lengths_from_ratios(class_ratios, num_labels)

            test_class_rule = get_single_class_rule(int(test_start_idx*num_labels), split_lengths["test"], sorted_label_set)
            traintest_dict["test"] = create_label_based_subset(dataset, labels, test_class_rule)
            split_lengths.pop("test", None)
            exclusion_rule = lambda label: not test_class_rule(label)
            sorted_label_set = sorted([x for x in sorted_label_set if exclusion_rule(x)])
            num_labels = len(sorted_label_set)

            start_idx = int((float(partition)/num_training_partitions)*num_labels)
            class_rules = get_class_rules(start_idx, split_lengths, sorted_label_set, exclusion_rule)
            for k, class_rule in class_rules.items():
                traintest_dict[k] = create_label_based_subset(dataset, labels, class_rule)

    return traintest_dict
