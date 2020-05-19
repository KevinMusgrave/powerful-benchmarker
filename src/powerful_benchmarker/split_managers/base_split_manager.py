#! /usr/bin/env python3

from collections import OrderedDict
import numpy as np
import torch
from ..utils import dataset_utils as d_u
import logging
import itertools
from collections import defaultdict

class BaseSplitManager:
    def __init__(self, hierarchy_level=0):
        self.hierarchy_level = hierarchy_level

    def assert_same_test_set_across_schemes(self):
        test_key = "test"
        prev_indices = None
        for (split_scheme_name, split_scheme) in self.split_schemes.items():
            for dataset_dict in split_scheme.values():
                curr_indices = np.array(dataset_dict[test_key].indices)
                if prev_indices is not None:
                    assert np.array_equal(curr_indices, prev_indices)
                prev_indices = curr_indices

    def assert_same_sets_in_both_train_and_eval(self):
        indices = defaultdict(dict)
        for (split_scheme_name, split_scheme) in self.split_schemes.items():
            for train_or_eval, dataset_dict in split_scheme.items():
                for split, dataset in dataset_dict.items():
                    indices[split][train_or_eval] = np.array(dataset.indices)
        for split in indices.values():
            prev_indices = None
            for train_or_eval in split.values():
                curr_indices = train_or_eval
                if prev_indices is not None:
                    assert np.array_equal(curr_indices, prev_indices)
                prev_indices = curr_indices


    def create_split_schemes(self, datasets):
        self.split_schemes = OrderedDict()
        self._create_split_schemes(datasets)
        self.split_assertions()
        self.split_scheme_names = list(self.split_schemes.keys())


    def _create_split_schemes(self, datasets):
        raise NotImplementedError

    def split_assertions(self):
        self.assert_same_test_set_across_schemes()
        self.assert_same_sets_in_both_train_and_eval()

    def set_curr_split_scheme(self, split_scheme_name):
        self.curr_split_scheme_name = split_scheme_name
        self.curr_split_scheme = self.split_schemes[self.curr_split_scheme_name]

    def get_dataset(self, train_or_eval, split_name, log_split_details=False):
        dataset = self.curr_split_scheme[train_or_eval][split_name]
        if log_split_details:
            logging.info("Getting split: {} / {} / length {} / using {} transform".format(self.curr_split_scheme_name, split_name, len(dataset), train_or_eval))
        return dataset

    def get_labels(self, *args, **kwargs):
        dataset = self.get_dataset(*args, **kwargs)
        return d_u.get_subset_dataset_labels(dataset)

    def get_num_labels(self, *args, **kwargs):
        labels = self.get_labels(*args, **kwargs)
        L = np.array(labels)
        L = d_u.get_labels_by_hierarchy(L, self.hierarchy_level)
        return len(set(L))

    def get_dataset_dict(self, train_or_eval, inclusion_list=None, exclusion_list=None):
        dataset_dict = {}
        curr_split_scheme = self.curr_split_scheme[train_or_eval]
        inclusion_list = list(curr_split_scheme.keys()) if inclusion_list is None else inclusion_list
        exclusion_list = [] if exclusion_list is None else exclusion_list
        allowed_list = [x for x in inclusion_list if x not in exclusion_list]
        for split_name, _ in curr_split_scheme.items():
            if split_name in allowed_list:
                dataset_dict[split_name] = self.get_dataset(train_or_eval, split_name, log_split_details=True)
        return dataset_dict
