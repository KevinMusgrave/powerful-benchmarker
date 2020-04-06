#! /usr/bin/env python3
import logging
logging.info("Importing packages in single_experiment_runner")
from easy_module_attribute_getter import YamlReader, PytorchGetter
from ..utils import common_functions as c_f, dataset_utils as d_u
import argparse
from .. import api_parsers
import glob
import os
logging.info("Done importing packages in single_experiment_runner")


class SingleExperimentRunner:
    def __init__(self, 
                dataset_root="datasets", 
                root_experiment_folder="experiments", 
                pytorch_home=None, 
                root_config_folder=None, 
                config_foldernames=None,
                global_db_path=None):
        self.dataset_root = dataset_root
        self.root_experiment_folder = root_experiment_folder
        self.global_db_path = global_db_path
        if pytorch_home is not None:
            os.environ["TORCH_HOME"] = pytorch_home
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
        from pytorch_metric_learning import trainers, losses, miners, regularizers, samplers, testers
        from .. import architectures
        from .. import datasets
        self.pytorch_getter = PytorchGetter(use_pretrainedmodels_package=True)
        self.pytorch_getter.register('model', architectures.misc_models)
        self.pytorch_getter.register('loss', losses)
        self.pytorch_getter.register('miner', miners)
        self.pytorch_getter.register('regularizer', regularizers)
        self.pytorch_getter.register('sampler', samplers)
        self.pytorch_getter.register('trainer', trainers)
        self.pytorch_getter.register('tester', testers)
        self.pytorch_getter.register('dataset', datasets)

    def set_YR(self):
        self.YR = self.setup_yaml_reader()


    def set_config_foldernames(self, config_foldernames=None):
        if config_foldernames:
            self.config_foldernames = config_foldernames
        else:
            self.config_foldernames = ["config_general", "config_models", "config_optimizers",
                                        "config_loss_and_miners", "config_transforms", "config_eval"]


    def run(self):
        if self.YR.args.reproduce_results:
            self.reproduce_results(self.YR)
        else:
            self.run_new_experiment(self.YR)

    def start_experiment(self, args):
        api_parser = getattr(api_parsers, "API"+args.training_method)(args, self.pytorch_getter, self.global_db_path)
        run_output = api_parser.run()
        self.eval_record_group_dicts = api_parser.get_eval_record_name_dict(return_all=True)
        del api_parser.tester_obj
        del api_parser.trainer
        del api_parser
        return run_output

    def run_new_experiment(self, YR):
        args, _, args.dict_of_yamls = YR.load_yamls(**self.determine_where_to_get_yamls(YR.args), max_merge_depth=float('inf'))
        return self.start_experiment(args)

    def reproduce_results(self, YR):
        configs_folder = os.path.join(YR.args.reproduce_results, 'configs')
        default_configs, experiment_configs = [], []
        for k in self.config_foldernames:
            default_configs.append(os.path.join(self.root_config_folder, k, "default.yaml")) #append default config file
        for k in self.config_foldernames:
            experiment_configs.append(os.path.join(configs_folder, "%s.yaml"%k)) #append experiment config file
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
                self.start_experiment(args)
                # start with a fresh set of args
                YR = self.setup_yaml_reader()
                # load the default configs, the experiment specific configs, plus the config diffs 
                for k in glob.glob(os.path.join(sub_folder, "*")):
                    experiment_configs.append(k)
                args, _, args.dict_of_yamls = YR.load_yamls(config_paths=default_configs+experiment_configs, max_merge_depth=0)
                # remove default configs from dict_of_yamls to avoid saving these as config diffs
                for d in default_configs:
                    args.dict_of_yamls.pop(d, None)
                args.resume_training = True
        return self.start_experiment(args)



    def setup_argparser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment_name", type=str, required=True)
        parser.add_argument("--resume_training", action="store_true")
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
            config_paths = [os.path.join(args.place_to_save_configs,'%s.yaml'%v) for v in self.config_foldernames]
        else:
            config_paths = []
            for subfolder in self.config_foldernames:
                for curr_yaml in getattr(args, subfolder):
                    config_paths.append(os.path.join(self.root_config_folder, subfolder, "%s.yaml"%curr_yaml))
        return {"config_paths": config_paths}