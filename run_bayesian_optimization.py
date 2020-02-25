from ax.service.ax_client import AxClient
from ax.service.utils.best_point import get_best_raw_objective_point
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements
from ax.plot.contour import interact_contour
from ax.modelbridge.registry import Models
import re
from easy_module_attribute_getter import utils as emag_utils
import run
import logging
import glob
import os
import csv
import pandas as pd
from utils import common_functions as c_f
logging.getLogger().setLevel(logging.INFO)

BAYESIAN_KEYWORDS=("~BAYESIAN~", "~LOG_BAYESIAN~", "~INT_BAYESIAN~")

def set_optimizable_params_and_bounds(args_dict, bayes_params, parent_key, keywords=BAYESIAN_KEYWORDS):
    for k, v in args_dict.items():
        if not isinstance(v, dict):
            for keyword in keywords:
                log_scale = "~LOG" in keyword
                value_type = "int" if "~INT" in keyword else "float"
                if k.endswith(keyword) and ("dict_of_yamls" not in parent_key):
                    assert isinstance(v, list)
                    actual_key = re.sub('\%s$'%keyword, '', k)
                    param_name = actual_key if parent_key == '' else "%s/%s"%(parent_key, actual_key)
                    bayes_params.append({"name": param_name, "type": "range", "bounds": v, "log_scale": log_scale, "value_type": value_type})
        else:
            next_parent_key = k if parent_key == '' else "%s/%s"%(parent_key, k)
            set_optimizable_params_and_bounds(v, bayes_params, next_parent_key, keywords)
    for keyword in keywords:
        emag_utils.remove_key_word(args_dict, keyword)

def read_yaml_and_find_bayes(config_foldernames):
    YR = run.setup_yaml_reader(config_foldernames)
    YR.args, _, YR.args.dict_of_yamls = YR.load_yamls(**run.determine_where_to_get_yamls(YR.args, config_foldernames), max_merge_depth=float('inf'))
    bayes_params = []
    set_optimizable_params_and_bounds(YR.args.__dict__, bayes_params, '')
    return YR, bayes_params

def replace_with_optimizer_values(param_path, input_dict, optimizer_value):
    for p in param_path.split("/"):
        if p in input_dict:
            if isinstance(input_dict[p], dict):
                input_dict = input_dict[p]
            else:
                input_dict[p] = optimizer_value

def get_latest_experiment_path(root_experiment_folder, experiment_name):
    experiment_number = len(glob.glob(os.path.join(root_experiment_folder, experiment_name+"*")))
    return os.path.join(root_experiment_folder, experiment_name)+str(experiment_number)

def run_experiment(config_foldernames, parameters):
    YR, _ = read_yaml_and_find_bayes(config_foldernames)
    for param_path, value in parameters.items():
        replace_with_optimizer_values(param_path, YR.args.__dict__, value)
        for sub_dict in YR.args.dict_of_yamls.values():
            replace_with_optimizer_values(param_path, sub_dict, value)
    YR.args.experiment_folder = get_latest_experiment_path(YR.args.root_experiment_folder, YR.args.experiment_name)
    YR.args.place_to_save_configs = os.path.join(YR.args.experiment_folder, "configs")
    return run.run(YR.args)

def get_log_folder(root_experiment_folder):
    return os.path.join(root_experiment_folder, "bayesian_optimizer_logs")

def get_all_log_paths(root_experiment_folder):
    return sorted(glob.glob(os.path.join(get_log_folder(root_experiment_folder), "log*.json")), reverse=True)

def save_new_log(ax_client, root_experiment_folder):
    log_paths = get_all_log_paths(root_experiment_folder)
    log_folder = get_log_folder(root_experiment_folder)
    c_f.makedir_if_not_there(log_folder)
    new_log_path = os.path.join(log_folder, "log%05d.json"%len(log_paths))
    ax_client.save_to_json_file(filepath=new_log_path)

def open_log(log_paths):
    for L in log_paths:
        try:
            ax_client = AxClient.load_from_json_file(filepath=L)
            break
        except:
            ax_client = None
    return ax_client

def get_ax_client(YR, bayes_params):
    log_paths = get_all_log_paths(YR.args.root_experiment_folder)
    ax_client = None
    if len(log_paths) > 0:
        ax_client = open_log(log_paths)
    if ax_client is None:
        ax_client = AxClient()
        ax_client.create_experiment(parameters=bayes_params, name=YR.args.experiment_name, minimize=False, objective_name=YR.args.eval_metric_for_best_epoch)
    return ax_client

def get_finished_experiment_names(root_experiment_folder):
    try:
        with open(os.path.join(root_experiment_folder, 'finished_experiment_names.csv'), 'r') as f:
            reader = csv.reader(f)
            output = list(reader)
    except:
        output = []
    return output

def write_finished_experiment_names(root_experiment_folder, experiment_name_list):
    with open(os.path.join(root_experiment_folder, 'finished_experiment_names.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(experiment_name_list)

def test_best_model(experiment_path):
    YR = run.setup_yaml_reader(config_foldernames)
    YR.args.evaluate = True
    YR.args.experiment_folder = experiment_path
    YR.args.place_to_save_configs = os.path.join(YR.args.experiment_folder, "configs")
    YR.args.splits_to_eval = ["test"]
    emag_utils.remove_key_word_recursively(YR.args.__dict__, "~OVERRIDE~")
    for keyword in BAYESIAN_KEYWORDS:
        emag_utils.remove_key_word_recursively(YR.args.__dict__, keyword)
    YR.args, _, YR.args.dict_of_yamls = YR.load_yamls(**run.determine_where_to_get_yamls(YR.args, config_foldernames), max_merge_depth=0)
    run.run(YR.args)

def plot_progress(ax_client, YR, root_experiment_folder, experiment_name):
    model = Models.GPEI(experiment=ax_client.experiment, data=ax_client.experiment.fetch_data())
    html_elements = []
    html_elements.append(plot_config_to_html(ax_client.get_optimization_trace()))
    try:
        html_elements.append(plot_config_to_html(interact_contour(model=model, metric_name=YR.args.eval_metric_for_best_epoch)))
    except:
        pass
    with open(os.path.join(root_experiment_folder, "optimization_plots.html"), 'w') as f:
        f.write(render_report_elements(experiment_name, html_elements))

if __name__ == "__main__":
    config_foldernames = ["config_general", "config_models", "config_optimizers",
                          "config_loss_and_miners", "config_transforms", "config_eval"]

    YR, bayes_params = read_yaml_and_find_bayes(config_foldernames)
    root_experiment_folder = YR.args.root_experiment_folder
    experiment_name = YR.args.experiment_name
    ax_client = get_ax_client(YR, bayes_params)
    num_explored_points = len(ax_client.experiment.trials) if ax_client.experiment.trials else 0
    finished_experiment_names = get_finished_experiment_names(root_experiment_folder)

    for i in range(num_explored_points, YR.args.bayesian_optimization_n_iter):
        logging.info("Optimization iteration %d"%i)
        experiment_path = get_latest_experiment_path(root_experiment_folder, experiment_name)
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=run_experiment(config_foldernames, parameters))
        save_new_log(ax_client, root_experiment_folder)
        finished_experiment_names.append([experiment_path])
        write_finished_experiment_names(root_experiment_folder, finished_experiment_names)
        plot_progress(ax_client, YR, root_experiment_folder, experiment_name)

    logging.info("DONE BAYESIAN OPTIMIZATION")
    df = ax_client.get_trials_data_frame()
    metric_column = pd.to_numeric(df[YR.args.eval_metric_for_best_epoch])
    best_trial_index = df['trial_index'].iloc[metric_column.idxmax()]
    best_experiment_path = finished_experiment_names[best_trial_index][0] 
    logging.info("BEST EXPERIMENT PATH: %s"%best_experiment_path)

    plot_progress(ax_client, YR, root_experiment_folder, experiment_name)
    best_parameters, best_values = get_best_raw_objective_point(ax_client.experiment)
    best_parameters_dict = {"best_experiment_path": best_experiment_path, 
                            "best_parameters": best_parameters, 
                            "best_values": {k:{"mean": float(v[0]), "SEM": float(v[1])} for k,v in best_values.items()}}
    c_f.write_yaml(os.path.join(root_experiment_folder, "best_parameters.yaml"), best_parameters_dict, open_as='w')

    test_best_model(best_experiment_path)