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

def experiment_filename(folder, basename, identifier, extension=".pth"):
    if identifier is None:
        return "%s/%s%s" % (folder, basename, extension)
    else:
        return "%s/%s_%s%s" % (folder, basename, str(identifier), extension)


def load_model(model_def, model_filename, device):
    try:
        model_def.load_state_dict(torch.load(model_filename, map_location=device))
    except BaseException:
        # original saved file with DataParallel
        state_dict = torch.load(model_filename)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model_def.load_state_dict(new_state_dict)


def save_model_or_optimizer(model, model_name, filepath):
    try:
        torch.save(model.cpu().state_dict(), filepath)
    except:
        torch.save(model.state_dict(), filepath)


def save_dict_of_models(input_dict, epoch, folder):
    for k, v in input_dict.items():
        opt_cond = "optimizer" in k
        if opt_cond or len([i for i in v.parameters()]) > 0:
            model_path = experiment_filename(folder, k, epoch)
            save_model_or_optimizer(v, k, model_path)


def load_dict_of_models(input_dict, resume_epoch, folder, device):
    for k, v in input_dict.items():
        opt_cond = "optimizer" in k
        if opt_cond or len([i for i in v.parameters()]) > 0:
            model_path = experiment_filename(folder, k, resume_epoch)
            logging.info("LOADING %s"%model_path)
            load_model(v, model_path, device)


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


def latest_version(folder, string_to_glob):
    items = glob.glob("%s/%s" % (folder, string_to_glob))
    if items == []:
        return None
    version = [int(x.split("_")[-1].split(".")[0]) for x in items]
    return max(version)


def latest_sub_experiment_epochs(sub_experiment_dir_dict):
    latest_epochs = {}
    for sub_experiment_name, folders in sub_experiment_dir_dict.items():
        model_folder = folders[0]
        latest_epochs[sub_experiment_name] = latest_version(model_folder, "/trunk_*.pth") or 0
    return latest_epochs


def get_all_resume_training_config_diffs(config_folder, split_scheme_names, num_variants_per_split_scheme):
    folder_base_name = "resume_training_config_diffs_"
    full_base_path = "%s/%s" % (config_folder, folder_base_name)
    config_diffs = sorted(glob.glob("%s*"%full_base_path))
    split_scheme_names = ["%s_%d"%(split_scheme,i) for (split_scheme,i) in list(itertools.product(split_scheme_names, range(num_variants_per_split_scheme)))]
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
        fname = '%s/%s.yaml' % (config_folder, config_category_name)
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
                new_dir = '%s/resume_training_config_diffs_'%(config_folder) + '_'.join([str(epoch) for epoch in latest_epochs])
                makedir_if_not_there(new_dir)
                fname = '%s/%s.yaml' % (new_dir, config_category_name)
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
