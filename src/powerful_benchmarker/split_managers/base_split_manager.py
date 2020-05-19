#! /usr/bin/env python3

from collections import OrderedDict
import numpy as np
import torch
from ..utils import dataset_utils as d_u
import logging
import itertools
from collections import defaultdict
from .split_scheme_holder import SplitSchemeHolder
import copy

class BaseSplitManager:
    def __init__(self, hierarchy_level=0):
        self.hierarchy_level = hierarchy_level
        self.split_scheme_holder = SplitSchemeHolder()

    def dataset_attribute_to_assert(self, dataset):
        return dataset.indices

    def assert_across(self, across_what, assertion, within_group=False, attribute_descriptor="indices", attribute_getter=None, **input_kwargs):
        if across_what == "split_scheme_names":
            names = self.split_scheme_holder.get_split_scheme_names()
        elif across_what == "transform_types":
            names = self.split_scheme_holder.get_transform_types()
        elif across_what == "split_names":
            names = self.split_scheme_holder.get_split_names()
        if attribute_getter is None:
            attribute_getter = self.dataset_attribute_to_assert
        datasets = []
        kwargs = copy.deepcopy(input_kwargs)
        for name in names:
            kwargs[across_what] = [name]
            datasets.append(self.split_scheme_holder.filter(**kwargs))
        if not within_group:
            datasets = zip(*datasets)
        for ds in datasets:
            for (x,y) in itertools.combinations(ds, 2):
                x_a, y_a = attribute_getter(x), attribute_getter(y)
                is_equal = np.array_equal(x_a, y_a)
                if assertion == "equal":
                    assert is_equal
                elif assertion == "not equal":
                    assert not is_equal
                elif assertion == "disjoint":
                    assert len(np.intersect1d(x_a, y_a)) == 0
                else:
                    raise ValueError('The assertion argument must be one of ["equal", "not_equal", "disjoint"]')
        input_kwargs_as_string = ", ".join(["{}={}".format(k, v) for k,v in input_kwargs.items()])
        splits = input_kwargs.pop("split_names")
        across_or_within = "across" if not within_group else "within"
        logging.info("Asserted: the {} set {} are {} {} {}".format(splits, attribute_descriptor, assertion, across_or_within, across_what))

    # datasets is two-level dictionary:
    # {train_transform: {train: dataset, val: dataset, test: dataset}, eval_transform:: {train: dataset, val: dataset, test: dataset}}
    def create_split_schemes(self, datasets):
        self.split_scheme_holder.set_split_schemes(self._create_split_schemes(datasets))
        self.split_assertions()
        
    def _create_split_schemes(self, datasets):
        raise NotImplementedError

    def split_assertions(self):
        for t_type in self.split_scheme_holder.get_transform_types():
            self.assert_across("split_scheme_names", "equal", transform_types=[t_type], split_names=["test"])
            self.assert_across("split_scheme_names", "not equal", transform_types=[t_type], split_names=["train"])
            self.assert_across("split_scheme_names", "disjoint", transform_types=[t_type], split_names=["val"])
            self.assert_across("split_scheme_names", "disjoint", within_group=True, transform_types=[t_type], split_names=self.split_scheme_holder.get_split_names())
        self.assert_across("transform_types", "equal", split_names=self.split_scheme_holder.get_split_names())

    def set_curr_split_scheme(self, split_scheme_name):
        self.split_scheme_holder.set_curr_split_scheme(split_scheme_name)

    def get_dataset(self, *args, **kwargs):
        return self.split_scheme_holder.get_dataset(*args, **kwargs)

    def get_labels(self, *args, **kwargs):
        dataset = self.get_dataset(*args, **kwargs)
        return d_u.get_subset_dataset_labels(dataset)

    def get_num_labels(self, *args, **kwargs):
        labels = self.get_labels(*args, **kwargs)
        return len(d_u.get_label_set(labels, self.hierarchy_level))

    def get_dataset_dict(self, *args, **kwargs):
        return self.split_scheme_holder.get_dataset_dict(*args, **kwargs)

    @property
    def split_scheme_names(self):
        return self.split_scheme_holder.get_split_scheme_names()

    @property
    def curr_split_scheme_name(self):
        return self.split_scheme_holder.curr_split_scheme_name
