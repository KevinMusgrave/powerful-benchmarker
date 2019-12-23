#! /usr/bin/env python3

from collections import OrderedDict

import numpy as np
import torch
from utils import dataset_utils as d_u
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
        self.create_split_schemes()

    def assert_splits_are_disjoint(self):
        for (split_scheme_name, split_scheme) in self.split_schemes.items():
            labels = []
            for split, (_, subset_indices) in split_scheme.items():
                labels.append(set(d_u.get_labels_by_hierarchy(self.original_dataset.labels[subset_indices], self.hierarchy_level)))
            for (x,y) in itertools.combinations(labels, 2):
                assert x.isdisjoint(y)

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
            self.assert_splits_are_disjoint()
        self.split_scheme_names = list(self.split_schemes.keys())

    def set_curr_split_scheme(self, split_scheme_name):
        self.curr_split_scheme_name = split_scheme_name
        self.curr_split_scheme = self.split_schemes[self.curr_split_scheme_name]

    def set_curr_split(self, split_name, is_training):
        """
        Sets self.dataset and self.labels to a specific split within the current
        split_scheme.
        """
        self.curr_split_name = split_name
        self.is_training = is_training
        transform = self.train_transform if self.is_training else self.eval_transform
        self.dataset, subset_indices = self.curr_split_scheme[self.curr_split_name]
        self.set_dataset_transform(transform)
        if self.is_training:
            label_source = self.original_dataset.labels[subset_indices]
            self.set_labels_to_indices(label_source)
            self.set_label_map()
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

    def set_labels_to_indices(self, labels):
        self.labels_to_indices = d_u.get_labels_to_indices(labels)

    def set_label_map(self):
        self.label_map = {}
        for hierarchy_level, v in self.labels_to_indices.items():
            self.label_map[hierarchy_level] = d_u.make_label_to_rank_dict(list(v.keys()))

    def map_labels(self, labels, hierarchy_level):
        return np.array([self.label_map[hierarchy_level][x] for x in labels], dtype=np.int)

    def get_num_labels(self, hierarchy_level):
        return len(self.labels_to_indices[hierarchy_level])

    def get_dataset_dict(self, exclusion_list, is_training):
        dataset_dict = {}
        for split_name, dataset in self.curr_split_scheme.items():
            if split_name not in exclusion_list:
                self.set_curr_split(split_name, is_training)
                dataset_dict[split_name] = self.dataset
        return dataset_dict
