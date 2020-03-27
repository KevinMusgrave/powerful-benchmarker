import logging
logging.info("Importing packages in bayes_opt_runner")
from ax.service.ax_client import AxClient
from ax.service.utils.best_point import get_best_raw_objective_point
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements
from ax.plot.contour import interact_contour
from ax.modelbridge.registry import Models
import re
from easy_module_attribute_getter import utils as emag_utils, YamlReader
import glob
import os
import csv
import pandas as pd
from ..utils import common_functions as c_f
from .single_experiment_runner import SingleExperimentRunner
import pytorch_metric_learning.utils.logging_presets as logging_presets
import math
from types import SimpleNamespace
import shutil
logging.info("Done importing packages in bayes_opt_runner")

BAYESIAN_KEYWORDS=("~BAYESIAN~", "~LOG_BAYESIAN~", "~INT_BAYESIAN~")


def set_optimizable_params_and_bounds(args_dict, bayes_params, parent_key, keywords=BAYESIAN_KEYWORDS):
    for k, v in args_dict.items():
        if not isinstance(v, dict):
            for keyword in keywords:
                log_scale = "~LOG" in keyword
                value_type = "int" if "~INT" in keyword else "float"
                if k.endswith(keyword) and ("dict_of_yamls" not in parent_key):
                    assert isinstance(v, list)
                    actual_key = re.sub('%s$'%keyword, '', k)
                    param_name = actual_key if parent_key == '' else "%s/%s"%(parent_key, actual_key)
                    bayes_params.append({"name": param_name, "type": "range", "bounds": v, "log_scale": log_scale, "value_type": value_type})
        else:
            next_parent_key = k if parent_key == '' else "%s/%s"%(parent_key, k)
            set_optimizable_params_and_bounds(v, bayes_params, next_parent_key, keywords)
    for keyword in keywords:
        emag_utils.remove_key_word(args_dict, keyword)



def replace_with_optimizer_values(param_path, input_dict, optimizer_value):
    for p in param_path.split("/"):
        if p in input_dict:
            if isinstance(input_dict[p], dict):
                input_dict = input_dict[p]
            else:
                input_dict[p] = optimizer_value


def open_log(log_paths):
    for L in log_paths:
        try:
            ax_client = AxClient.load_from_json_file(filepath=L)
            break
        except:
            ax_client = None
    return ax_client


def remove_keywords(YR):
    emag_utils.remove_key_word_recursively(YR.args.__dict__, "~OVERRIDE~")
    for keyword in BAYESIAN_KEYWORDS:
        emag_utils.remove_key_word_recursively(YR.args.__dict__, keyword)


class BayesOptRunner(SingleExperimentRunner):
    def __init__(self, bayes_opt_iters, num_reproductions, **kwargs):
        super().__init__(**kwargs)
        self.bayes_opt_iters = bayes_opt_iters
        self.num_reproductions = num_reproductions
        self.experiment_name = self.YR.args.experiment_name
        self.bayes_opt_root_experiment_folder = os.path.join(self.root_experiment_folder, self.experiment_name)
        if self.global_db_path is None:
            self.global_db_path = os.path.join(self.bayes_opt_root_experiment_folder, "bayes_opt_experiments.db")
        self.csv_folder = os.path.join(self.bayes_opt_root_experiment_folder, "bayes_opt_record_keeper_logs")
        self.tensorboard_folder = os.path.join(self.bayes_opt_root_experiment_folder, "bayes_opt_tensorboard_logs")
        self.ax_log_folder = os.path.join(self.bayes_opt_root_experiment_folder, "bayes_opt_ax_logs")
        self.best_parameters_filename = os.path.join(self.bayes_opt_root_experiment_folder, "best_parameters.yaml")
        self.most_recent_parameters_filename = os.path.join(self.bayes_opt_root_experiment_folder, "most_recent_parameters.yaml")
        self.bayes_opt_table_name = "bayes_opt"

    def set_YR(self):
        self.YR, self.bayes_params = self.read_yaml_and_find_bayes()


    def run(self):    
        ax_client = self.get_ax_client()
        num_explored_points = len(ax_client.experiment.trials) if ax_client.experiment.trials else 0
        is_new_experiment = num_explored_points == 0
        record_keeper, _, _ = logging_presets.get_record_keeper(self.csv_folder, self.tensorboard_folder)

        for i in range(num_explored_points, self.bayes_opt_iters):
            logging.info("Optimization iteration %d"%i)
            sub_experiment_name = self.get_sub_experiment_name(i)
            parameters, trial_index, experiment_func = self.get_parameters_and_trial_index(ax_client, sub_experiment_name)
            ax_client.complete_trial(trial_index=trial_index, raw_data=experiment_func(parameters, sub_experiment_name))
            self.save_new_log(ax_client)
            self.update_records(record_keeper, ax_client, i)
            self.plot_progress(ax_client)

        logging.info("DONE BAYESIAN OPTIMIZATION")
        # self.plot_progress(ax_client)
        best_sub_experiment_name = self.save_best_parameters(record_keeper, ax_client)
        self.test_model(best_sub_experiment_name)
        self.reproduce_results(best_sub_experiment_name)


    def get_parameters_and_trial_index(self, ax_client, sub_experiment_name):
        if os.path.isdir(self.get_sub_experiment_path(sub_experiment_name)):
            recent = c_f.load_yaml(self.most_recent_parameters_filename)
            parameters, trial_index = recent["parameters"], recent["trial_index"]
            ax_client.attach_trial(parameters)
            experiment_func = self.resume_training
        else:
            parameters, trial_index = ax_client.get_next_trial()
            c_f.write_yaml(self.most_recent_parameters_filename, {"parameters": parameters, "trial_index": trial_index}, open_as='w')
            experiment_func = self.run_new_experiment
        return parameters, trial_index, experiment_func


    def get_sub_experiment_name(self, iteration):
        return self.experiment_name+str(iteration)
        

    def get_sub_experiment_path(self, sub_experiment_name):
        return os.path.join(self.bayes_opt_root_experiment_folder, sub_experiment_name)


    def get_sub_experiment_bayes_opt_filename(self, sub_experiment_path):
        return os.path.join(sub_experiment_path, "bayes_opt_parameters.yaml")


    def get_all_log_paths(self):
        return sorted(glob.glob(os.path.join(self.ax_log_folder, "log*.json")), reverse=True)


    def save_new_log(self, ax_client):
        log_paths = self.get_all_log_paths()
        log_folder = self.ax_log_folder
        c_f.makedir_if_not_there(log_folder)
        new_log_path = os.path.join(log_folder, "log%05d.json"%len(log_paths))
        ax_client.save_to_json_file(filepath=new_log_path)

    def update_records(self, record_keeper, ax_client, iteration):
        df_as_dict = ax_client.get_trials_data_frame().to_dict()
        most_recent = {k.replace('/','_'):v[iteration] for k,v in df_as_dict.items()}
        record_keeper.update_records(most_recent, global_iteration=iteration, input_group_name_for_non_objects=self.bayes_opt_table_name)
        record_keeper.save_records()

    def save_best_parameters(self, record_keeper, ax_client):
        q = record_keeper.query("SELECT * FROM {0} WHERE {1}=(SELECT max({1}) FROM {0})".format(self.bayes_opt_table_name, self.YR.args.eval_primary_metric))[0]
        best_trial_index = int(q['trial_index'])
        best_sub_experiment_name = self.get_sub_experiment_name(best_trial_index)
        best_sub_experiment_path = self.get_sub_experiment_path(best_sub_experiment_name) 
        logging.info("BEST SUB EXPERIMENT NAME: %s"%best_sub_experiment_name)

        best_parameters, best_values = get_best_raw_objective_point(ax_client.experiment)
        assert math.isclose(best_values[self.YR.args.eval_primary_metric][0], q[self.YR.args.eval_primary_metric])
        best_parameters_dict = {"best_sub_experiment_name": best_sub_experiment_name,
                                "best_sub_experiment_path": best_sub_experiment_path, 
                                "best_parameters": best_parameters, 
                                "best_values": {k:{"mean": float(v[0]), "SEM": float(v[1])} for k,v in best_values.items()}}
        c_f.write_yaml(self.best_parameters_filename, best_parameters_dict, open_as='w')
        return best_sub_experiment_name

    def get_ax_client(self):
        log_paths = self.get_all_log_paths()
        ax_client = None
        if len(log_paths) > 0:
            ax_client = open_log(log_paths)
        if ax_client is None:
            ax_client = AxClient()
            ax_client.create_experiment(parameters=self.bayes_params, name=self.YR.args.experiment_name, minimize=False, objective_name=self.YR.args.eval_primary_metric)
        return ax_client


    def plot_progress(self, ax_client):
        model = Models.GPEI(experiment=ax_client.experiment, data=ax_client.experiment.fetch_data())
        html_elements = []
        html_elements.append(plot_config_to_html(ax_client.get_optimization_trace()))
        try:
            html_elements.append(plot_config_to_html(interact_contour(model=model, metric_name=self.YR.args.eval_primary_metric)))
        except:
            pass
        with open(os.path.join(self.bayes_opt_root_experiment_folder, "optimization_plots.html"), 'w') as f:
            f.write(render_report_elements(self.experiment_name, html_elements))


    def read_yaml_and_find_bayes(self):
        YR = self.setup_yaml_reader()
        YR.args, _, YR.args.dict_of_yamls = YR.load_yamls(**self.determine_where_to_get_yamls(YR.args), max_merge_depth=float('inf'))
        bayes_params = []
        set_optimizable_params_and_bounds(YR.args.__dict__, bayes_params, '')
        return YR, bayes_params


    def set_experiment_name_and_place_to_save_configs(self, YR):
        YR.args.experiment_folder = self.get_sub_experiment_path(YR.args.experiment_name)
        YR.args.place_to_save_configs = os.path.join(YR.args.experiment_folder, "configs")


    def get_simplified_yaml_reader(self, experiment_name):
        YR = YamlReader()
        YR.args, _ = self.setup_argparser().parse_known_args() # we want to ignore the "unknown" args in this case
        YR.args.dataset_root = self.dataset_root
        YR.args.experiment_name = experiment_name
        self.set_experiment_name_and_place_to_save_configs(YR)
        return YR


    def delete_sub_experiment_folder(self, sub_experiment_name):
        logging.warning("Deleting and starting fresh for %s"%sub_experiment_name)
        shutil.rmtree(self.get_sub_experiment_path(sub_experiment_name))
        global_record_keeper, _, _ = logging_presets.get_record_keeper(self.csv_folder, self.tensorboard_folder, self.global_db_path, sub_experiment_name, False)
        global_record_keeper.record_writer.global_db.delete_experiment(sub_experiment_name)


    def try_resuming(self, YR):
        try:
            output = super().run_new_experiment(YR)
        except Exception as e:
            logging.error(repr(e))
            logging.warning("Could not resume training for %s"%YR.args.experiment_name)
            self.delete_sub_experiment_folder(YR.args.experiment_name)
            output = None
        return output


    def resume_training(self, parameters, sub_experiment_name):
        local_YR = self.get_simplified_yaml_reader(sub_experiment_name)
        local_YR.args.resume_training = True

        try:
            loaded_parameters = c_f.load_yaml(self.get_sub_experiment_bayes_opt_filename(local_YR.args.experiment_folder))
            assert parameters == loaded_parameters
            parameter_load_successful = True
        except Exception as e:
            logging.error(repr(e))
            logging.warning("Input parameters and loaded parameters don't match for %s"%sub_experiment_name)
            self.delete_sub_experiment_folder(sub_experiment_name)
            parameter_load_successful = False

        output = self.try_resuming(local_YR) if parameter_load_successful else None
        return output if output is not None else self.run_new_experiment(parameters, sub_experiment_name)


    def run_new_experiment(self, parameters, sub_experiment_name):
        local_YR, _ = self.read_yaml_and_find_bayes()
        for param_path, value in parameters.items():
            replace_with_optimizer_values(param_path, local_YR.args.__dict__, value)
            for sub_dict in local_YR.args.dict_of_yamls.values():
                replace_with_optimizer_values(param_path, sub_dict, value)
        local_YR.args.experiment_name = sub_experiment_name
        self.set_experiment_name_and_place_to_save_configs(local_YR)
        c_f.makedir_if_not_there(local_YR.args.experiment_folder)
        c_f.write_yaml(self.get_sub_experiment_bayes_opt_filename(local_YR.args.experiment_folder), parameters, open_as='w')
        return self.start_experiment(local_YR.args)


    def test_model(self, sub_experiment_name):
        local_YR = self.get_simplified_yaml_reader(sub_experiment_name)
        local_YR.args.evaluate = True
        local_YR.args.splits_to_eval = ["test"]
        for meta_testing_method in [None, "ConcatenateEmbeddings"]:
            local_YR.args.__dict__["meta_testing_method~OVERRIDE~"] = meta_testing_method
            super().run_new_experiment(local_YR)


    def reproduce_results(self, sub_experiment_name):
        for i in range(self.num_reproductions):
            local_YR = self.get_simplified_yaml_reader("%s_reproduction%d"%(sub_experiment_name, i))
            local_YR.args.reproduce_results = self.get_sub_experiment_path(sub_experiment_name)
            output = None
            if os.path.isdir(local_YR.args.experiment_folder):
                local_YR.args.resume_training = True
                output = self.try_resuming(local_YR)
            if output is None:
                super().reproduce_results(local_YR)
            self.test_model(local_YR.args.experiment_name)
