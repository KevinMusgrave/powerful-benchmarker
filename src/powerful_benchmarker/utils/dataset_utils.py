#! /usr/bin/env python3

import torch.utils.data

def get_subset_dataset_labels(subset_dataset):
    return subset_dataset.dataset.labels[subset_dataset.indices]

def get_labels_by_hierarchy(labels, hierarchy_level):
    if labels.ndim == 2:
        labels = labels[:, hierarchy_level]
    return labels

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
    return torch.utils.data.Subset(dataset, idx_to_keep)


def create_rule_based_subset(dataset, rule_input, rule):
    idx_to_keep = [i for i, r in enumerate(rule_input) if rule(r)]
    return create_subset(dataset, idx_to_keep)


def wrapped_range_rule(a, b, exclusion_rule=None):
    exclusion_rule = (lambda input_scalar: True) if exclusion_rule is None else exclusion_rule

    if a < b:
        range_rule = lambda input_scalar: (a <= input_scalar <= b)
    elif a > b:
        range_rule = lambda input_scalar: (input_scalar >= a or input_scalar <= b)
    return lambda input_scalar: (range_rule(input_scalar) and exclusion_rule(input_scalar))


def get_wrapped_range(start_idx, length_of_range, length_of_list):
    s = start_idx % length_of_list
    e = (start_idx + length_of_range - 1) % length_of_list
    return s, e

def get_single_wrapped_range_rule(start_idx, split_length, list_for_rule, exclusion_rule=None):
    s, e = get_wrapped_range(start_idx, split_length, len(list_for_rule))
    s = list_for_rule[s]
    e = list_for_rule[e]
    return wrapped_range_rule(s, e, exclusion_rule)

def get_wrapped_range_rules(start_idx, split_lengths, list_for_rule, exclusion_rule=None):
    rules = {}
    for k, v in split_lengths.items():
        rules[k] = get_single_wrapped_range_rule(start_idx, v, list_for_rule, exclusion_rule)
        start_idx += v
    return rules

def split_lengths_from_ratios(ratios, total_length):
    split_lengths = {k: int(total_length * v) for k, v in ratios.items()}
    split_lengths["train"] += total_length - sum(v for k, v in split_lengths.items())
    assert sum(v for _, v in split_lengths.items()) == total_length
    return split_lengths


