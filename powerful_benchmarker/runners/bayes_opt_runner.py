import logging
logging.info("Importing packages in bayes_opt_runner")
from ax.service.ax_client import AxClient
from ax.service.utils.best_point import get_best_raw_objective_point
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements
from ax.plot.contour import interact_contour
from ax.modelbridge.registry import Models
import re
from easy_module_attribute_getter import utils as emag_utils
import glob
import os
import csv
import pandas as pd
from ..utils import common_functions as c_f
from .single_experiment_runner import SingleExperimentRunner
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


class BayesOptRunner(SingleExperimentRunner):
    def __init__(self, bayes_opt_iters, **kwargs):
        super().__init__(**kwargs)
        self.bayes_opt_iters = bayes_opt_iters
        self.experiment_name = self.YR.args.experiment_name
        self.bayes_opt_root_experiment_folder = os.path.join(self.root_experiment_folder, self.experiment_name)
        self.finished_csv_filename = "finished_sub_experiment_names.csv"
        

    def set_YR(self):
        self.YR, self.bayes_params = self.read_yaml_and_find_bayes()


    def run(self):    
        ax_client = self.get_ax_client()
        num_explored_points = len(ax_client.experiment.trials) if ax_client.experiment.trials else 0
        finished_sub_experiment_names = self.get_finished_sub_experiment_names()

        for i in range(num_explored_points, self.bayes_opt_iters):
            logging.info("Optimization iteration %d"%i)
            sub_experiment_name = self.get_latest_sub_experiment_name()
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=self.run_new_experiment(parameters, sub_experiment_name))
            self.save_new_log(ax_client)
            finished_sub_experiment_names.append([sub_experiment_name])
            self.write_finished_sub_experiment_names(finished_sub_experiment_names)
            self.plot_progress(ax_client)

        logging.info("DONE BAYESIAN OPTIMIZATION")
        df = ax_client.get_trials_data_frame()
        metric_column = pd.to_numeric(df[self.YR.args.eval_primary_metric])
        best_trial_index = df['trial_index'].iloc[metric_column.idxmax()]
        best_sub_experiment_name = finished_sub_experiment_names[best_trial_index][0]
        best_sub_experiment_path = self.get_sub_experiment_path(best_sub_experiment_name) 
        logging.info("BEST SUB EXPERIMENT NAME: %s"%best_sub_experiment_name)

        self.plot_progress(ax_client)
        best_parameters, best_values = get_best_raw_objective_point(ax_client.experiment)
        best_parameters_dict = {"best_sub_experiment_name": best_sub_experiment_name,
                                "best_sub_experiment_path": best_sub_experiment_path, 
                                "best_parameters": best_parameters, 
                                "best_values": {k:{"mean": float(v[0]), "SEM": float(v[1])} for k,v in best_values.items()}}
        c_f.write_yaml(os.path.join(self.bayes_opt_root_experiment_folder, "best_parameters.yaml"), best_parameters_dict, open_as='w')

        self.test_best_model(best_sub_experiment_name)


    def run_new_experiment(self, parameters, sub_experiment_name):
        local_YR, _ = self.read_yaml_and_find_bayes()
        for param_path, value in parameters.items():
            replace_with_optimizer_values(param_path, local_YR.args.__dict__, value)
            for sub_dict in local_YR.args.dict_of_yamls.values():
                replace_with_optimizer_values(param_path, sub_dict, value)
        local_YR.args.experiment_name = sub_experiment_name
        local_YR.args.experiment_folder = self.get_sub_experiment_path(sub_experiment_name)
        local_YR.args.place_to_save_configs = os.path.join(local_YR.args.experiment_folder, "configs")
        return self.start_experiment(local_YR.args)


    def get_latest_sub_experiment_name(self):
        experiment_number = len(glob.glob(os.path.join(self.bayes_opt_root_experiment_folder, self.experiment_name+"*")))
        return self.experiment_name+str(experiment_number)
        

    def get_sub_experiment_path(self, sub_experiment_name):
        return os.path.join(self.bayes_opt_root_experiment_folder, sub_experiment_name)


    def get_log_folder(self):
        return os.path.join(self.bayes_opt_root_experiment_folder, "bayesian_optimizer_logs")


    def get_all_log_paths(self):
        return sorted(glob.glob(os.path.join(self.get_log_folder(), "log*.json")), reverse=True)


    def save_new_log(self, ax_client):
        log_paths = self.get_all_log_paths()
        log_folder = self.get_log_folder()
        c_f.makedir_if_not_there(log_folder)
        new_log_path = os.path.join(log_folder, "log%05d.json"%len(log_paths))
        ax_client.save_to_json_file(filepath=new_log_path)


    def get_ax_client(self):
        log_paths = self.get_all_log_paths()
        ax_client = None
        if len(log_paths) > 0:
            ax_client = open_log(log_paths)
        if ax_client is None:
            ax_client = AxClient()
            ax_client.create_experiment(parameters=self.bayes_params, name=self.YR.args.experiment_name, minimize=False, objective_name=self.YR.args.eval_primary_metric)
        return ax_client


    def get_finished_sub_experiment_names(self):
        try:
            with open(os.path.join(self.bayes_opt_root_experiment_folder, self.finished_csv_filename), 'r') as f:
                reader = csv.reader(f)
                output = list(reader)
        except:
            output = []
        return output


    def write_finished_sub_experiment_names(self, experiment_name_list):
        with open(os.path.join(self.bayes_opt_root_experiment_folder, self.finished_csv_filename), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(experiment_name_list)


    def test_best_model(self, sub_experiment_name):
        local_YR = self.setup_yaml_reader()
        local_YR.args.evaluate = True
        local_YR.args.experiment_name = sub_experiment_name
        local_YR.args.experiment_folder = self.get_sub_experiment_path(sub_experiment_name)
        local_YR.args.place_to_save_configs = os.path.join(local_YR.args.experiment_folder, "configs")
        local_YR.args.splits_to_eval = ["test"]
        emag_utils.remove_key_word_recursively(local_YR.args.__dict__, "~OVERRIDE~")
        for keyword in BAYESIAN_KEYWORDS:
            emag_utils.remove_key_word_recursively(local_YR.args.__dict__, keyword)
        local_YR.args, _, local_YR.args.dict_of_yamls = local_YR.load_yamls(**self.determine_where_to_get_yamls(local_YR.args), max_merge_depth=0)
        self.start_experiment(local_YR.args)


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


    def reproduce_results(self):
        pass
