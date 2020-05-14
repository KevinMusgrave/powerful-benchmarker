#! /usr/bin/env python3
import logging
logging.info("Importing packages in single_experiment_runner")
from ..utils import common_functions as c_f, dataset_utils as d_u
from easy_module_attribute_getter import utils as emag_utils
from .base_runner import BaseRunner
import glob
import os
logging.info("Done importing packages in single_experiment_runner")


class SingleExperimentRunner(BaseRunner):

    def run(self):
        if self.YR.args.reproduce_results:
            self.reproduce_results(self.YR)
        else:
            self.run_new_experiment_or_resume(self.YR)

    def start_experiment(self, args):
        api_parser = self.get_api_parser(args)
        run_output = api_parser.run()
        del api_parser.tester_obj
        del api_parser.trainer
        del api_parser
        return run_output

    def run_new_experiment_or_resume(self, YR):
        merge_argparse = self.merge_argparse_when_resuming if YR.args.resume_training else True
        args, _, args.dict_of_yamls = YR.load_yamls(self.determine_where_to_get_yamls(YR.args), 
                                                    max_merge_depth=float('inf'), 
                                                    merge_argparse=merge_argparse)
        return self.start_experiment(args)

    def reproduce_results(self, YR, starting_fresh_hook=None):
        configs_folder = os.path.join(YR.args.reproduce_results, 'configs')
        all_config_paths = self.get_root_config_paths(YR.args)
        experiment_config_paths = self.get_saved_config_paths(configs_folder)
        for k, v in experiment_config_paths.items():
            all_config_paths[k].extend(v)
        args, _, args.dict_of_yamls = YR.load_yamls(config_paths=all_config_paths, 
                                                    max_merge_depth=0, 
                                                    merge_argparse=self.merge_argparse_when_resuming)

        # check if there were config diffs if training was resumed
        if YR.args.special_split_scheme_name:
            base_split_name = YR.args.special_split_scheme_name
        else:
            base_split_name = d_u.get_base_split_name(YR.args.test_size, YR.args.test_start_idx, YR.args.num_training_partitions)
        resume_training_dict = c_f.get_all_resume_training_config_diffs(configs_folder, base_split_name)

        if len(resume_training_dict) > 0:
            for sub_folder, num_epochs_dict in resume_training_dict.items():
                # train until the next config diff was made
                args.num_epochs_train = num_epochs_dict
                self.start_experiment(args)
                # Start fresh
                YR = self.setup_yaml_reader()
                if starting_fresh_hook: starting_fresh_hook(YR)
                # load the default configs, the experiment specific configs, plus the config diffs 
                for k in glob.glob(os.path.join(sub_folder, "*")):
                    config_name = os.path.splitext(os.path.basename(k))[0]
                    all_config_paths[config_name].append(k)
                args, _, args.dict_of_yamls = YR.load_yamls(config_paths=all_config_paths, 
                                                            max_merge_depth=0, 
                                                            merge_argparse=self.merge_argparse_when_resuming)
                args.resume_training = "latest"
        return self.start_experiment(args)