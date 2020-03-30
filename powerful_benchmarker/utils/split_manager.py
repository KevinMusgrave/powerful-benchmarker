#! /usr/bin/env python3

from collections import OrderedDict

import numpy as np
import torch
from . import dataset_utils as d_u
import logging
import itertools

class SplitManager:
    def __init__(
        self,
        dataset=None,
        train_transform=None,
        eval_transform=None,
        test_size=None,
        test_start_idx=None,
        num_training_partitions=2,
        num_training_sets=1,
        special_split_scheme_name=None,
        hierarchy_level=0
    ):
        self.original_dataset = dataset
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.test_size = test_size
        self.test_start_idx = test_start_idx
        self.num_training_partitions = num_training_partitions
        self.num_training_sets = num_training_sets
        self.special_split_scheme_name = special_split_scheme_name
        self.hierarchy_level = hierarchy_level
        self.is_training = True
        self.create_split_schemes()

    def assert_splits_are_class_disjoint(self):
        for (split_scheme_name, split_scheme) in self.split_schemes.items():
            labels = []
            for split, dataset in split_scheme.items():
                labels.append(set(d_u.get_labels_by_hierarchy(self.original_dataset.labels[dataset.indices], self.hierarchy_level)))
            for (x,y) in itertools.combinations(labels, 2):
                assert x.isdisjoint(y)

    def assert_same_test_set_across_schemes(self):
        test_key = "val" if self.special_split_scheme_name == "old_approach" else "test"
        prev_indices = None
        for (split_scheme_name, split_scheme) in self.split_schemes.items():
            curr_indices = np.array(split_scheme[test_key].indices)
            if prev_indices is not None:
                assert np.array_equal(curr_indices, prev_indices)
            prev_indices = curr_indices


    def create_split_schemes(self):
        """
        Creates a dictionary where each key is the name of a split_scheme.
        Each value is a dictionary with "train", "val", "test" keys, corresponding
        to subsets of self.original_dataset
        """
        self.split_schemes = OrderedDict()
        if self.special_split_scheme_name:
            self.split_schemes[self.special_split_scheme_name] = d_u.create_one_split_scheme(self.original_dataset, 
                                                                                            scheme_name=self.special_split_scheme_name)
        else:
            for partition in range(self.num_training_sets):
                name = d_u.get_base_split_name(self.test_size, self.test_start_idx, self.num_training_partitions, partition=partition)
                self.split_schemes[name] = d_u.create_one_split_scheme(self.original_dataset, 
                                                                        partition=partition,
                                                                        num_training_partitions=self.num_training_partitions,
                                                                        test_size=self.test_size, 
                                                                        test_start_idx=self.test_start_idx,
                                                                        hierarchy_level=self.hierarchy_level)
        if self.special_split_scheme_name != "predefined": 
            self.assert_splits_are_class_disjoint()
            self.assert_same_test_set_across_schemes()
        self.split_scheme_names = list(self.split_schemes.keys())
        self.split_names = list(self.split_schemes[self.split_scheme_names[0]].keys())

    def set_curr_split_scheme(self, split_scheme_name):
        self.curr_split_scheme_name = split_scheme_name
        self.curr_split_scheme = self.split_schemes[self.curr_split_scheme_name]

    def set_curr_split(self, split_name, is_training, log_split_details=False):
        """
        Sets self.dataset and self.labels to a specific split within the current
        split_scheme.
        """
        self.curr_split_name = split_name
        self.is_training = is_training
        transform = self.train_transform if self.is_training else self.eval_transform
        self.dataset = self.curr_split_scheme[self.curr_split_name]
        self.set_dataset_transform(transform)
        if self.is_training:
            self.labels = self.original_dataset.labels[self.dataset.indices]
        if log_split_details:
            logging.info("SPLIT: %s / %s / length %d" % (self.curr_split_scheme_name, self.curr_split_name, len(self.dataset)))

    def set_transforms(self, train_transform, eval_transform):
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        if self.is_training:
            self.set_dataset_transform(self.train_transform)
        else:
            self.set_dataset_transform(self.eval_transform)

    def set_dataset_transform(self, transform):
        self.dataset.dataset.transform = transform

    def get_num_labels(self):
        self.set_curr_split("train", True)
        L = np.array(self.labels)
        if L.ndim == 2:
            L = L[:, self.hierarchy_level]
        return len(set(L))

    def get_dataset_dict(self, inclusion_list=None, exclusion_list=None, is_training=False):
        logging.info("COLLECTING DATASETS FOR EVAL")
        dataset_dict = {}
        inclusion_list = list(self.curr_split_scheme.keys()) if inclusion_list is None else inclusion_list
        exclusion_list = [] if exclusion_list is None else exclusion_list
        allowed_list = [x for x in inclusion_list if x not in exclusion_list]
        for split_name, _ in self.curr_split_scheme.items():
            if split_name in allowed_list:
                self.set_curr_split(split_name, is_training, log_split_details=True)
                dataset_dict[split_name] = self.dataset
        return dataset_dict
