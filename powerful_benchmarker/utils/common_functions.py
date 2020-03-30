#! /usr/bin/env python3

import collections
import errno
import glob
import os
import itertools

import numpy as np
import torch
from torch.autograd import Variable
import yaml
from easy_module_attribute_getter import utils as emag_utils
import logging
import inspect
import pytorch_metric_learning.utils.common_functions as pml_cf
import datetime
import sqlite3


def move_optimizer_to_gpu(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def makedir_if_not_there(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_yaml(fname):
    with open(fname, 'r') as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml


def write_yaml(fname, input_dict, open_as):
    with open(fname, open_as) as outfile:
        yaml.dump(input_dict, outfile, default_flow_style=False, sort_keys=False)


def latest_sub_experiment_epochs(sub_experiment_dir_dict):
    latest_epochs = {}
    for sub_experiment_name, folders in sub_experiment_dir_dict.items():
        model_folder = folders[0]
        latest_epochs[sub_experiment_name] = pml_cf.latest_version(model_folder, "trunk_*.pth") or 0
    return latest_epochs


def get_all_resume_training_config_diffs(config_folder, split_scheme_name, num_training_sets):
    folder_base_name = "resume_training_config_diffs_"
    full_base_path = os.path.join(config_folder, folder_base_name)
    config_diffs = sorted(glob.glob("%s*"%full_base_path))
    if num_training_sets > 1:
        split_scheme_names = ["%s%d"%(split_scheme,i) for (split_scheme,i) in list(itertools.product(split_scheme_name, range(num_training_sets)))]
    else:
        split_scheme_names = [split_scheme_name]
    resume_training_dict = {}
    for k in config_diffs:
        latest_epochs = [int(x) for x in k.replace(full_base_path,"").split('_')]
        resume_training_dict[k] = {split_scheme:epoch for (split_scheme,epoch) in zip(split_scheme_names,latest_epochs)}
    return resume_training_dict


def save_config_files(config_folder, dict_of_yamls, resume_training, reproduce_results, latest_epochs):
    makedir_if_not_there(config_folder)
    new_dir = None
    for k, v in dict_of_yamls.items():
        k_split = k.split('/')
        config_category_name = k_split[-2] 
        if not config_category_name.startswith('config_'):
            config_category_name = os.path.splitext(k_split[-1])[0]
        fname = os.path.join(config_folder, '%s.yaml'%config_category_name)
        if not resume_training:
            if os.path.isfile(fname):
                v = emag_utils.merge_two_dicts(load_yaml(fname), v, max_merge_depth=0 if reproduce_results else float('inf'))
            write_yaml(fname, v, 'w')
        else:
            curr_yaml = load_yaml(fname)
            yaml_diff = {}
            for k2, v2 in v.items():
                if (k2 not in curr_yaml) or (v2 != curr_yaml[k2]):
                    yaml_diff[k2] = v2
            if yaml_diff != {}:
                new_dir = os.path.join(config_folder, 'resume_training_config_diffs_' + '_'.join([str(epoch) for epoch in latest_epochs]))
                makedir_if_not_there(new_dir)
                fname = os.path.join(new_dir, '%s.yaml' %config_category_name)
                write_yaml(fname, yaml_diff, 'a')

def get_last_linear(input_model, return_name=False):
    for name in ["fc", "last_linear"]:
        last_layer = getattr(input_model, name, None)
        if last_layer:
            if return_name:
                return last_layer, name
            return last_layer

def set_last_linear(input_model, set_to):
    setattr(input_model, get_last_linear(input_model, return_name=True)[1], set_to)


def check_init_arguments(input_obj, str_to_check):
    obj_stack = [input_obj]
    while len(obj_stack) > 0:
        curr_obj = obj_stack.pop()
        obj_stack += list(curr_obj.__bases__)
        if str_to_check in str(inspect.signature(curr_obj.__init__)):
            return True
    return False


def try_getting_db_count(record_keeper, table_name):
    try:
        len_of_existing_record = record_keeper.query("SELECT count(*) FROM %s"%table_name, use_global_db=False)[0]["count(*)"] 
    except sqlite3.OperationalError:
        len_of_existing_record = 0
    return len_of_existing_record


def get_datetime():
    return datetime.datetime.now()