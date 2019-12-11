#! /usr/bin/env python3
from easy_module_attribute_getter import YamlReader
import argparse
import api_parsers
import logging
logging.getLogger().setLevel(logging.INFO)

def setup_argparser(config_foldernames):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_experiment_folder", type=str,
                        default="/home/tkm45/NEW_STUFF/experiments")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--run_eval_only", nargs="+", type=str, default=None)
    parser.add_argument("--splits_to_eval", nargs="+", type=str, default=None)
    parser.add_argument("--reproduce_results", type=str, default=None)
    parser.add_argument("--root_config_folder", type=str, required=False, default="configs")
    for c in config_foldernames:
        parser.add_argument("--%s" % c, nargs="+", type=str, required=False, default=["default"])
    return parser

def determine_where_to_get_yamls(args, config_foldernames):
    if args.resume_training or args.run_eval_only:
        config_paths = ['%s/%s.yaml'%(args.place_to_save_configs, v) for v in config_foldernames]
    else:
        config_paths = []
        for subfolder in config_foldernames:
            for curr_yaml in getattr(args, subfolder):
                config_paths.append("%s/%s/%s.yaml"%(args.root_config_folder, subfolder, curr_yaml))
    return {"config_paths": config_paths}

def run(args):
    api_parser = getattr(api_parsers, "API"+args.training_method)(args)
    api_parser.run()

def reproduce_results(YR, config_foldernames):
    configs_folder = '%s/configs'%YR.args.reproduce_results
    starting_config_paths = []
    for k in config_foldernames:
        starting_config_paths.append("%s/%s/default.yaml"%(YR.args.root_config_folder, k)) #append default config file
        starting_config_paths.append('%s/%s.yaml'%(configs_folder, k)) #append experiment config file
    args, _, args.dict_of_yamls = YR.load_yamls(config_paths=starting_config_paths)
    run(args)

def run_new_experiment(YR, config_foldernames):
    args, _, args.dict_of_yamls = YR.load_yamls(**determine_where_to_get_yamls(YR.args, config_foldernames), max_merge_depth=float('inf'))
    run(args)

if __name__ == "__main__":
    config_foldernames = ["config_general", "config_models", "config_optimizers",
                          "config_loss_and_miners", "config_transforms", "config_eval"]

    argparser = setup_argparser(config_foldernames)

    YR = YamlReader(argparser=argparser)
    YR.args.experiment_folder = "%s/%s" % (YR.args.root_experiment_folder, YR.args.experiment_name)
    YR.args.place_to_save_configs = "%s/%s" % (YR.args.experiment_folder, "configs")

    if YR.args.reproduce_results:
        reproduce_results(YR, config_foldernames)
    else:
        run_new_experiment(YR, config_foldernames)