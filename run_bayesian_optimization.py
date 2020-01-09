from ax.service.ax_client import AxClient
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements
import re
from easy_module_attribute_getter import utils as emag_utils
import run
import logging
import glob
import os
from utils import common_functions as c_f
logging.getLogger().setLevel(logging.INFO)

def set_optimizable_params_and_bounds(args_dict, bayes_params, parent_key, keywords=("~BAYESIAN~", "~LOG_BAYESIAN~", "~INT_BAYESIAN~")):
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


def run_experiment(config_foldernames, parameters):
    YR, _ = read_yaml_and_find_bayes(config_foldernames)
    for param_path, value in parameters.items():
        replace_with_optimizer_values(param_path, YR.args.__dict__, value)
        for sub_dict in YR.args.dict_of_yamls.values():
            replace_with_optimizer_values(param_path, sub_dict, value)
    experiment_number = len(glob.glob("%s/%s*"%(YR.args.root_experiment_folder, YR.args.experiment_name)))
    YR.args.experiment_folder = "%s/%s%d" % (YR.args.root_experiment_folder, YR.args.experiment_name, experiment_number)
    YR.args.place_to_save_configs = "%s/%s" % (YR.args.experiment_folder, "configs")
    return run.run(YR.args)

def get_log_path(root_experiment_folder):
    return "%s/bayesian_optimization_logs.json"%(root_experiment_folder)

def get_ax_client(YR, bayes_params):
    log_path = get_log_path(YR.args.root_experiment_folder)
    if os.path.isfile(log_path):
        ax_client = AxClient.load_from_json_file(filepath=log_path)
    else:
        ax_client = AxClient()
        ax_client.create_experiment(parameters=bayes_params, name=YR.args.experiment_name, minimize=False, objective_name=YR.args.eval_metric_for_best_epoch)
    return ax_client, log_path

def plot_progress(ax_client, root_experiment_folder, experiment_name):
    html_elements = []
    html_elements.append(plot_config_to_html(ax_client.get_optimization_trace()))
    try:
        html_elements.append(plot_config_to_html(ax_client.get_contour_plot()))
    except:
        pass
    with open("%s/optimization_plots.html"%root_experiment_folder, 'w') as f:
        f.write(render_report_elements(experiment_name, html_elements))

if __name__ == "__main__":
    config_foldernames = ["config_general", "config_models", "config_optimizers",
                          "config_loss_and_miners", "config_transforms", "config_eval"]

    YR, bayes_params = read_yaml_and_find_bayes(config_foldernames)
    ax_client, log_path = get_ax_client(YR, bayes_params)
    num_explored_points = len(ax_client.experiment.trials) if ax_client.experiment.trials else 0
    n_iter = YR.args.bayesian_optimization_n_iter - num_explored_points

    for i in range(num_explored_points, n_iter):
        logging.info("Optimization iteration %d"%i)
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=run_experiment(config_foldernames, parameters))
        ax_client.save_to_json_file(filepath=log_path)
        plot_progress(ax_client, YR.args.root_experiment_folder, YR.args.experiment_name)


    logging.info("DONE BAYESIAN OPTIMIZATION")
    best_parameters, best_values = ax_client.get_best_parameters()
    best_parameters_dict = {"best_parameters": best_parameters, "means": best_values[0]}
    c_f.write_yaml("%s/best_parameters.yaml"%YR.args.root_experiment_folder, best_parameters_dict, open_as='w')