#! /usr/bin/env python3
import logging
logging.info("Importing packages in base_runner")
from easy_module_attribute_getter import YamlReader, PytorchGetter
from ..utils import common_functions as c_f
import argparse
import glob
import os
from collections import defaultdict
logging.info("Done importing packages in base_runner")


class BaseRunner:
    def __init__(self, 
                dataset_root="datasets", 
                root_experiment_folder="experiments", 
                pytorch_home=None, 
                root_config_folder=None, 
                config_foldernames=None,
                global_db_path=None,
                merge_argparse_when_resuming=False):
        self.dataset_root = dataset_root
        self.root_experiment_folder = root_experiment_folder
        self.global_db_path = global_db_path
        self.merge_argparse_when_resuming = merge_argparse_when_resuming
        self.pytorch_home = pytorch_home
        if pytorch_home is not None:
            os.environ["TORCH_HOME"] = self.pytorch_home
        if root_config_folder is not None:
            self.root_config_folder = root_config_folder
        else:
            self.root_config_folder = os.path.join(os.path.dirname(__file__), "../configs")
        self.init_pytorch_getter()
        self.set_config_foldernames(config_foldernames)
        self.set_YR()
        
    def register(self, module_type, module):
        self.pytorch_getter.register(module_type, module)

    def init_pytorch_getter(self):
        from pytorch_metric_learning import trainers, losses, miners, regularizers, samplers, testers, utils
        from .. import architectures
        from .. import datasets
        from .. import api_parsers
        self.pytorch_getter = PytorchGetter(use_pretrainedmodels_package=True)
        self.pytorch_getter.register('model', architectures.misc_models)
        self.pytorch_getter.register('loss', losses)
        self.pytorch_getter.register('miner', miners)
        self.pytorch_getter.register('regularizer', regularizers)
        self.pytorch_getter.register('sampler', samplers)
        self.pytorch_getter.register('trainer', trainers)
        self.pytorch_getter.register('tester', testers)
        self.pytorch_getter.register('dataset', datasets)
        self.pytorch_getter.register('api_parser', api_parsers)
        self.pytorch_getter.register('accuracy_calculator', utils.AccuracyCalculator)


    def set_YR(self):
        self.YR = self.setup_yaml_reader()


    def set_config_foldernames(self, config_foldernames=None):
        if config_foldernames:
            self.config_foldernames = config_foldernames
        else:
            self.config_foldernames = ["config_general", "config_models", "config_optimizers",
                                        "config_loss_and_miners", "config_transforms", "config_eval"]


    def run(self):
        raise NotImplementedError


    def get_api_parser(self, args):
        api_parser_kwargs = {"args": args, "pytorch_getter": self.pytorch_getter, "global_db_path": self.global_db_path}
        return self.pytorch_getter.get('api_parser', class_name="API"+args.training_method, params=api_parser_kwargs)

    def setup_argparser(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--experiment_name", type=str, required=True)
        parser.add_argument("--resume_training", type=str, default=None, choices=["latest", "best"])
        parser.add_argument("--evaluate", action="store_true")
        parser.add_argument("--splits_to_eval", nargs="+", type=str, default=["val"])
        parser.add_argument("--reproduce_results", type=str, default=None)
        for c in self.config_foldernames:
            parser.add_argument("--%s" % c, nargs="+", type=str, required=False, default=["default"])
        return parser


    def setup_yaml_reader(self):
        argparser = self.setup_argparser()
        YR = YamlReader(argparser=argparser)
        YR.args.dataset_root = self.dataset_root
        YR.args.experiment_folder = os.path.join(self.root_experiment_folder, YR.args.experiment_name)
        YR.args.place_to_save_configs = os.path.join(YR.args.experiment_folder, "configs")
        return YR


    def determine_where_to_get_yamls(self, args):
        if args.resume_training or args.evaluate:
            config_paths = self.get_saved_config_paths(args.place_to_save_configs)
        else:
            config_paths = self.get_root_config_paths(args)
        return config_paths


    def get_saved_config_paths(self, config_location):
        return {v: [os.path.join(config_location,'%s.yaml'%v)] for v in self.config_foldernames}


    def get_root_config_paths(self, args):
        config_paths = defaultdict(list)
        for subfolder in self.config_foldernames:
            for curr_yaml in getattr(args, subfolder):
                with_subfolder = os.path.join(self.root_config_folder, subfolder, "%s.yaml"%curr_yaml)
                without_subfolder = os.path.join(self.root_config_folder, "%s.yaml"%subfolder)
                if os.path.isfile(with_subfolder):
                    config_paths[subfolder].append(with_subfolder)
                else:
                    config_paths[subfolder].append(without_subfolder)
        return config_paths