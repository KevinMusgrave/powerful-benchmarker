from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
import re
from easy_module_attribute_getter import utils as emag_utils
import argparse
import run
import logging
import glob
logging.getLogger().setLevel(logging.INFO)

def get_optimizable_params_and_bounds(args_dict, bayes_params, parent_key, keywords=("~BAYESIAN~","~LOG_BAYESIAN~")):
    for k, v in args_dict.items():
        if not isinstance(v, dict):
            for keyword in keywords:
                if k.endswith(keyword) and ("dict_of_yamls" not in parent_key):
                    assert isinstance(v, list)
                    param_name = k if parent_key == '' else "%s/%s"%(parent_key, k)
                    bayes_params[param_name] = v
        else:
            next_parent_key = k if parent_key == '' else "%s/%s"%(parent_key, k)
            get_optimizable_params_and_bounds(v, bayes_params, next_parent_key, keywords)
            for keyword in keywords:
                emag_utils.remove_key_word(v, keyword)
    for keyword in keywords:
        emag_utils.remove_key_word(args_dict, keyword)


def read_yaml_and_find_bayes(config_foldernames):
    YR = run.setup_yaml_reader(config_foldernames)
    YR.args, _, YR.args.dict_of_yamls = YR.load_yamls(**run.determine_where_to_get_yamls(YR.args, config_foldernames), max_merge_depth=float('inf'))
    bayes_params = {}
    get_optimizable_params_and_bounds(YR.args.__dict__, bayes_params, '')
    return YR, bayes_params

def replace_with_optimizer_values(param_path, input_dict, optimizer_value):
    curr_dict = input_dict
    for p in param_path:
        actual_key = p
        for keyword, function in [("~BAYESIAN~", lambda x: float(x)), ("~LOG_BAYESIAN~", lambda x: float(10**x))]:
            if actual_key.endswith(keyword):
                actual_key = re.sub('\%s$'%keyword, '', actual_key)
                conversion = function
        if actual_key in curr_dict:
            if isinstance(curr_dict[actual_key], dict):
                curr_dict = curr_dict[actual_key]
            else:
                curr_dict[actual_key] = conversion(optimizer_value)


def run_bayesian_optimization(config_foldernames):
    def rbo(**kwargs):
        YR, _ = read_yaml_and_find_bayes(config_foldernames)
        for key, value in kwargs.items():
            param_path = key.split("/")
            replace_with_optimizer_values(param_path, YR.args.__dict__, value)
            for sub_dict in YR.args.dict_of_yamls.values():
                replace_with_optimizer_values(param_path, sub_dict, value)
        experiment_number = len(glob.glob("%s/%s*"%(YR.args.root_experiment_folder, YR.args.experiment_name)))
        YR.args.experiment_folder = "%s/%s%d" % (YR.args.root_experiment_folder, YR.args.experiment_name, experiment_number)
        YR.args.place_to_save_configs = "%s/%s" % (YR.args.experiment_folder, "configs")
        return run.run(YR.args)
    return rbo

def get_bayesian_logger_paths(root_experiment_folder):
    base_filename = "bayesian_optimization_logs"
    existing_logs = glob.glob("%s/%s*.json"%(root_experiment_folder, base_filename))
    new_log_path = "%s/%s%d.json"%(root_experiment_folder, base_filename, len(existing_logs))
    return new_log_path, existing_logs

if __name__ == "__main__":
    config_foldernames = ["config_general", "config_models", "config_optimizers",
                          "config_loss_and_miners", "config_transforms", "config_eval"]

    YR, bayes_params = read_yaml_and_find_bayes(config_foldernames)
    bayesian_logger_path, existing_logger_paths = get_bayesian_logger_paths(YR.args.root_experiment_folder)
    optimizer = BayesianOptimization(f=run_bayesian_optimization(config_foldernames), pbounds=bayes_params)
    if len(existing_logger_paths) > 0:
        load_logs(optimizer, logs=existing_logger_paths)
        logging.info("LOADED PREVIOUS BAYESIAN OPTIMIZER LOGS")
    bayesian_logger = JSONLogger(path=bayesian_logger_path)
    optimizer.subscribe(Events.OPTMIZATION_STEP, bayesian_logger)

    num_explored_points = len(optimizer._space)
    init_points = max(0, YR.args.bayesian_optimization_init_points - num_explored_points)
    n_iter = YR.args.bayesian_optimization_n_iter - num_explored_points
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    logging.info("DONE BAYESIAN OPTIMIZATION")
    logging.info("BEST RESULT:")
    logging.info(optimizer.max)