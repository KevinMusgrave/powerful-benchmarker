#! /usr/bin/env python3
from easy_module_attribute_getter import YamlReader
from utils import common_functions as c_f, dataset_utils as d_u
import argparse
import api_parsers
import logging
import glob
logging.getLogger().setLevel(logging.INFO)

def setup_argparser(config_foldernames):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_home", type=str, default="/home/tkm45/NEW_STUFF/pytorch_models")
    parser.add_argument("--dataset_root", type=str, default="/scratch")
    parser.add_argument("--root_experiment_folder", type=str, default="/home/tkm45/NEW_STUFF/experiments")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--splits_to_eval", nargs="+", type=str, default=["train", "val"])
    parser.add_argument("--reproduce_results", type=str, default=None)
    parser.add_argument("--root_config_folder", type=str, required=False, default="configs")
    for c in config_foldernames:
        parser.add_argument("--%s" % c, nargs="+", type=str, required=False, default=["default"])
    return parser

def setup_yaml_reader(config_foldernames):
    argparser = setup_argparser(config_foldernames)
    YR = YamlReader(argparser=argparser)
    YR.args.experiment_folder = "%s/%s" % (YR.args.root_experiment_folder, YR.args.experiment_name)
    YR.args.place_to_save_configs = "%s/%s" % (YR.args.experiment_folder, "configs")
    return YR

def determine_where_to_get_yamls(args, config_foldernames):
    if args.resume_training or args.evaluate:
        config_paths = ['%s/%s.yaml'%(args.place_to_save_configs, v) for v in config_foldernames]
    else:
        config_paths = []
        for subfolder in config_foldernames:
            for curr_yaml in getattr(args, subfolder):
                config_paths.append("%s/%s/%s.yaml"%(args.root_config_folder, subfolder, curr_yaml))
    return {"config_paths": config_paths}

def run(args):
    api_parser = getattr(api_parsers, "API"+args.training_method)(args)
    return api_parser.run()

def reproduce_results(YR, config_foldernames):
    configs_folder = '%s/configs'%YR.args.reproduce_results
    default_configs, experiment_configs = [], []
    for k in config_foldernames:
        default_configs.append("%s/%s/default.yaml"%(YR.args.root_config_folder, k)) #append default config file
    for k in config_foldernames:
        experiment_configs.append('%s/%s.yaml'%(configs_folder, k)) #append experiment config file
    args, _, args.dict_of_yamls = YR.load_yamls(config_paths=default_configs+experiment_configs, max_merge_depth=0)

    # check if there were config diffs if training was resumed
    if YR.args.special_split_scheme_name:
        num_training_sets = 1
        base_split_name = YR.args.special_split_scheme_name
    else:
        num_training_sets = YR.args.num_training_sets
        base_split_name = d_u.get_base_split_name(YR.args.test_size, YR.args.test_start_idx, YR.args.num_training_partitions)
    resume_training_dict = c_f.get_all_resume_training_config_diffs(configs_folder, base_split_name, num_training_sets)
    if len(resume_training_dict) > 0:
        for sub_folder, num_epochs_dict in resume_training_dict.items():
            # train until the next config diff was made
            args.num_epochs_train = num_epochs_dict
            run(args)
            # start with a fresh set of args
            YR = setup_yaml_reader(config_foldernames)
            # load the default configs, the experiment specific configs, plus the config diffs 
            for k in glob.glob("%s/*"%sub_folder):
                experiment_configs.append(k)
            args, _, args.dict_of_yamls = YR.load_yamls(config_paths=default_configs+experiment_configs, max_merge_depth=0)
            # remove default configs from dict_of_yamls to avoid saving these as config diffs
            for d in default_configs:
                args.dict_of_yamls.pop(d, None)
            args.resume_training = True
    run(args)

def run_new_experiment(YR, config_foldernames):
    args, _, args.dict_of_yamls = YR.load_yamls(**determine_where_to_get_yamls(YR.args, config_foldernames), max_merge_depth=float('inf'))
    return run(args)


if __name__ == "__main__":
    config_foldernames = ["config_general", "config_models", "config_optimizers",
                          "config_loss_and_miners", "config_transforms", "config_eval"]

    YR = setup_yaml_reader(config_foldernames)
   
    if YR.args.reproduce_results:
        reproduce_results(YR, config_foldernames)
    else:
        run_new_experiment(YR, config_foldernames)