import glob
import json
import os

import h5py
import numpy as np
import pandas as pd
import tqdm
from pytorch_adapt.utils import common_functions as c_f

from powerful_benchmarker.utils.constants import TRIALS_FILENAME

from .constants import VALIDATOR_TESTS_FOLDER


def get_condition_fn(validator_name, validator_args_str, trial_range):
    trial_range_specified = trial_range != []
    if trial_range_specified:
        trial_range = np.arange(*trial_range)

    def fn(iteration, folder):
        filepath = get_df_filepath(folder, validator_name, validator_args_str)
        if os.path.isfile(filepath):
            try:
                df = pd.read_pickle(filepath)
                if len(df) > 0:
                    return False
                return True
            except:  # in case it's corrupted or something
                return True
        if trial_range_specified and iteration not in trial_range:
            return False
        return True

    return fn


def get_df_filepath(folder, validator_name, validator_args_str):
    filename = os.path.join(folder, VALIDATOR_TESTS_FOLDER)
    c_f.makedir_if_not_there(filename)
    filename = os.path.join(filename, validator_str(validator_name, validator_args_str))
    return f"{filename}.pkl"


def dict_to_str(x):
    return json.dumps(x, sort_keys=True)


def get_exp_folders(folder, name):
    exp_path = os.path.join(folder, name)
    if not os.path.isdir(exp_path):
        return []
    trials_filename = os.path.join(exp_path, TRIALS_FILENAME)
    with open(trials_filename, "r") as f:
        trials_info = pd.read_csv(f)
    trials_info = trials_info[trials_info["state"] == "COMPLETE"]
    trial_nums = trials_info["number"]
    return [os.path.join(exp_path, str(x)) for x in trial_nums]


def read_exp_config_file(folder):
    config_file = os.path.join(folder, "configs", "args_and_trial_params.json")
    with open(config_file, "r") as f:
        exp_config = json.load(f)
    exp_config["exp_validator"] = exp_config.pop("validator")
    return exp_config


def apply_to_data(exp_folders, condition, fn=None, end_fn=None):
    for i, e in enumerate(exp_folders):
        if not condition(i, e):
            continue
        if fn:
            print(e)
            exp_config = read_exp_config_file(e)
            features_file = os.path.join(e, "features", "features.hdf5")
            with h5py.File(features_file, "r") as data:
                for k in tqdm.tqdm(data.keys()):
                    fn(k, data[k], exp_config, e)
        if end_fn:
            end_fn(e)


# str representation of dict as input
def validator_args_underscore_delimited(validator_args_str):
    return "_".join([f"{k}_{v}" for k, v in json.loads(validator_args_str).items()])


def validator_str(validator_name, validator_args_str):
    v_str = validator_args_underscore_delimited(validator_args_str)
    return f"{validator_name}_{v_str}"


def count_pkls(exp_folder, validator, fn=None):
    exp_paths = sorted(glob.glob(f"{exp_folder}/*"))
    num_pkls = 0
    for p in exp_paths:
        if os.path.isdir(p):
            exps = glob.glob(f"{p}/*")
            for e in exps:
                if os.path.isdir(e):
                    pkls = os.path.join(e, VALIDATOR_TESTS_FOLDER, f"{validator}*.pkl")
                    pkls = glob.glob(pkls)
                    num_pkls += len(pkls)
                    if fn:
                        fn(pkls)
    print("num_pkls", num_pkls)
