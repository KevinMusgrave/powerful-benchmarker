#! /usr/bin/env python3

from collections import OrderedDict

import numpy as np
from utils import dataset_utils as d_u
import logging

class SplitManager:
    def __init__(
        self,
        dataset=None,
        train_transform=None,
        eval_transform=None,
        split_scheme_names=None,
        num_variants_per_split_scheme=1,
        input_dataset_splits=None,
    ):
        self.original_dataset = dataset
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.split_scheme_names = split_scheme_names
        self.num_variants_per_split_scheme = num_variants_per_split_scheme
        self.input_dataset_splits = input_dataset_splits
        self.create_split_schemes()

    def create_split_schemes(self):
        """
        Creates a dictionary where each key is the name of a split_scheme.
        Each value is a dictionary with "train", "val", "test" keys, corresponding
        to subsets of self.original_dataset
        """
        if self.input_dataset_splits is not None:
            self.split_schemes = self.input_dataset_splits
            self.split_scheme_names = list(self.split_schemes.keys())
        else:
            self.split_schemes = OrderedDict()
            for split_scheme_name in self.split_scheme_names:
                for i,start_idx in enumerate(np.linspace(0, 1, self.num_variants_per_split_scheme, endpoint=False)):
                    self.split_schemes['%s_%d'%(split_scheme_name,i)] = d_u.create_one_split_scheme(
                        self.original_dataset, split_scheme_name, start_idx)
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
            if self.input_dataset_splits is None:
                label_source = self.original_dataset.labels[subset_indices]
            else:
                label_source = self.dataset.labels
            self.set_labels_to_indices(label_source)
            self.set_label_map()
        logging.info("SPLIT: %s / %s / length %d" %
              (self.curr_split_scheme_name, self.curr_split_name, len(self.dataset)))

    def set_transforms(self, train_transform, eval_transform):
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        if self.is_training:
            self.set_dataset_transform(self.train_transform)
        else:
            self.set_dataset_transform(self.eval_transform)

    def set_dataset_transform(self, transform):
        if self.input_dataset_splits is None:
            self.dataset.dataset.transform = transform
        else:
            self.dataset.transform = transform  

    def set_labels_to_indices(self, labels):
        self.labels_to_indices = d_u.get_labels_to_indices(labels)

    def set_label_map(self):
        self.label_map = {}
        for hierarchy_level, v in self.labels_to_indices.items():
            self.label_map[hierarchy_level] = d_u.make_label_to_rank_dict(list(v.keys()))

    def map_labels(self, labels, hierarchy_level):
        try:
            output = {}
            for k, v in labels.items():
                if v is None:
                    output[k] = None
                else:
                    output[k] = np.array([self.label_map[hierarchy_level][x] for x in v], dtype=np.int)
            return output
        except BaseException:
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
