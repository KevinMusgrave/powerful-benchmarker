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
        api_parser_kwargs = {"args": args, "pytorch_getter": self.pytorch_getter, "global_db_path": self.global_db_path}
        api_parser = self.pytorch_getter.get('api_parser', class_name="API"+args.training_method, params=api_parser_kwargs)
        run_output = api_parser.run()
        self.eval_record_group_dicts = api_parser.get_eval_record_name_dict(return_all=True)
        del api_parser.tester_obj
        del api_parser.trainer
        del api_parser
        return run_output

    def run_new_experiment_or_resume(self, YR):
        merge_argparse = self.merge_argparse_when_resuming if YR.args.resume_training else True
        args, _, args.dict_of_yamls = YR.load_yamls(**self.determine_where_to_get_yamls(YR.args), 
                                                    max_merge_depth=float('inf'), 
                                                    merge_argparse=self.merge_argparse_when_resuming)
        return self.start_experiment(args)

    def reproduce_results(self, YR):
        configs_folder = os.path.join(YR.args.reproduce_results, 'configs')
        default_configs = self.get_root_config_paths(YR.args)
        experiment_configs = self.get_saved_config_paths(configs_folder)
        print("default_configs", default_configs)
        print("experiment_configs", experiment_configs)
        args, _, args.dict_of_yamls = YR.load_yamls(config_paths=default_configs+experiment_configs, 
                                                    max_merge_depth=0, 
                                                    merge_argparse=self.merge_argparse_when_resuming)

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
                # remove default configs from dict_of_yamls to avoid saving these as config diffs
                for d in default_configs:
                    args.dict_of_yamls.pop(d, None)
                self.start_experiment(args)
                # Start fresh
                YR = self.setup_yaml_reader()
                # But remove nested dictionaries from the command line. 
                # In other words, even if merge_argparse_when_resuming is True, nested dictionaries will not be included. 
                emag_utils.remove_dicts(YR.args.__dict__)
                # load the default configs, the experiment specific configs, plus the config diffs 
                for k in glob.glob(os.path.join(sub_folder, "*")):
                    experiment_configs.append(k)
                args, _, args.dict_of_yamls = YR.load_yamls(config_paths=default_configs+experiment_configs, 
                                                            max_merge_depth=0, 
                                                            merge_argparse=self.merge_argparse_when_resuming)
                args.resume_training = "latest"
        # remove default configs from dict_of_yamls to avoid saving these as config diffs
        for d in default_configs:
            args.dict_of_yamls.pop(d, None)
        return self.start_experiment(args)