from collections import OrderedDict
from ..utils import common_functions as c_f
import logging

class SplitSchemeHolder:
    def __init__(self):
        self.split_schemes = OrderedDict()

    def filter(self, split_scheme_names=None, transform_types=None, split_names=None):
        output_list = []
        if split_scheme_names is None:
            split_scheme_names = self.get_split_scheme_names()
        if transform_types is None:
            transform_types = self.get_transform_types()
        if split_names is None:
            split_names = self.get_split_names()
        for k1, v1 in self.split_schemes.items():
            if k1 not in split_scheme_names:
                continue
            for k2, v2 in v1.items():
                if k2 not in transform_types:
                    continue
                for k3, v3 in v2.items():
                    if k3 not in split_names:
                        continue
                    output_list.append(v3)
        return output_list
        

    def set_split_schemes(self, input_split_schemes):
        self.split_schemes = input_split_schemes
        self.set_curr_split_scheme(self.get_split_scheme_names()[0])

    def set_curr_split_scheme(self, split_scheme_name):
        self.curr_split_scheme_name = split_scheme_name
        self.curr_split_scheme = self.split_schemes[self.curr_split_scheme_name]

    def get_split_scheme_names(self):
        return list(self.split_schemes.keys())

    def get_transform_types(self):
        return list(c_f.first_val_of_dict(self.split_schemes).keys())

    def get_split_names(self):
        return list(c_f.first_val_of_dict(c_f.first_val_of_dict(self.split_schemes)).keys())

    def get_dataset(self, transform_type, split_name, log_split_details=False):
        dataset = self.curr_split_scheme[transform_type][split_name]
        if log_split_details:
            logging.info("Getting split: {} / {} / length {} / using {} transform".format(self.curr_split_scheme_name, split_name, len(dataset), transform_type))
        return dataset

    def get_dataset_dict(self, transform_type, inclusion_list=None, exclusion_list=None):
        dataset_dict = {}
        curr_split_scheme = self.curr_split_scheme[transform_type]
        inclusion_list = list(curr_split_scheme.keys()) if inclusion_list is None else inclusion_list
        exclusion_list = [] if exclusion_list is None else exclusion_list
        allowed_list = [x for x in inclusion_list if x not in exclusion_list]
        for split_name, _ in curr_split_scheme.items():
            if split_name in allowed_list:
                dataset_dict[split_name] = self.get_dataset(transform_type, split_name, log_split_details=True)
        return dataset_dict