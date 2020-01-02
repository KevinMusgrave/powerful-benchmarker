from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
import re
from easy_module_attribute_getter import utils as emag_utils
import argparse
import run
import logging
import glob
logging.getLogger().setLevel(logging.INFO)

def get_optimizable_params_and_bounds(args_dict, bayes_params, parent_key, bayes_keyword="~BAYESIAN~"):
    for k, v in args_dict.items():
        if not isinstance(v, dict):
            if k.endswith(bayes_keyword):
                actual_key = re.sub('\%s$'%bayes_keyword, '', k)
                assert isinstance(v, list)
                bayes_params["%s/%s"%(parent_key,actual_key)] = v
        else:
            next_parent_key = k if parent_key == '' else "%s/%s"%(parent_key, k)
            get_optimizable_params_and_bounds(v, bayes_params, next_parent_key, bayes_keyword)
            emag_utils.remove_key_word(v, bayes_keyword)
    emag_utils.remove_key_word(args_dict, bayes_keyword)


def read_yaml_and_find_bayes(config_foldernames):
    YR = run.setup_yaml_reader(config_foldernames)
    bayes_params = {}
    get_optimizable_params_and_bounds(YR.args.__dict__, bayes_params, '')
    return YR, bayes_params


def run_bayesian_optimization(config_foldernames):
    def rbo(**kwargs):
        YR, _ = read_yaml_and_find_bayes(config_foldernames)
        for key, value in kwargs.items():
            param_path = key.split("/")
            curr_dict = YR.args.__dict__
            for p in param_path:
                if not isinstance(curr_dict[p], dict):
                    curr_dict[p] = float(value)
                else:
                    curr_dict = curr_dict[p]
        experiment_number = len(glob.glob("%s/%s*"%(YR.args.root_experiment_folder, YR.args.experiment_name)))
        YR.args.experiment_folder = "%s/%s%d" % (YR.args.root_experiment_folder, YR.args.experiment_name, experiment_number)
        YR.args.place_to_save_configs = "%s/%s" % (YR.args.experiment_folder, "configs")
        return run.run_new_experiment(YR, config_foldernames)
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
    try:
        load_logs(optimizer, logs=existing_logger_paths)
        logging.info("LOADED PREVIOUS BAYESIAN OPTIMIZER LOGS")
    except:
        pass
    bayesian_logger = JSONLogger(path=bayesian_logger_path)
    optimizer.subscribe(Events.OPTMIZATION_STEP, bayesian_logger)

    optimizer.maximize(init_points=YR.args.bayesian_optimization_init_points, n_iter=YR.args.bayesian_optimization_n_iter)

    logging.info("DONE BAYESIAN OPTIMIZATION")
    logging.info("BEST RESULT:")
    logging.info(optimizer.max)