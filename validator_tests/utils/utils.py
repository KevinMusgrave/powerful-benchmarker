import glob
import json
import os

import h5py
import numpy as np
import pandas as pd
import tqdm
from pytorch_adapt.utils import common_functions as c_f

from .constants import VALIDATOR_TESTS_FOLDER

SPLIT_NAMES = ["src_train", "src_val", "target_train", "target_val"]
AVERAGE_NAMES = ["micro", "macro"]


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
    trials_filename = os.path.join(exp_path, "trials.csv")
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


def process_from_hdf5(x):
    if isinstance(x, bytes):
        return str(x, "utf-8")
    if isinstance(x, np.ndarray) and isinstance(x[0], bytes):
        return [str(y, "utf-8") for y in x]
    return x


def is_not_features(k):
    return all(not k.startswith(f"{x}_") for x in SPLIT_NAMES)


def exp_specific_columns(df):
    exclude = ["score", "validator", "validator_args", *all_acc_score_column_names()]
    return [x for x in df.columns.values if x not in exclude]


def acc_score_column_name(split, average):
    return f"{split}_{average}"


def all_acc_score_column_names():
    return [acc_score_column_name(x, y) for x in SPLIT_NAMES for y in AVERAGE_NAMES]


def get_acc_rows(df, split, average):
    args = dict_to_str({"average": average, "split": split})
    return df[(df["validator_args"] == args) & (df["validator"] == "Accuracy")]


def get_acc_df(df, split, average):
    df = get_acc_rows(df, split, average)
    df = df.drop(columns=["validator", "validator_args"])
    return df.rename(columns={"score": acc_score_column_name(split, average)})


def get_all_acc(df):
    output = None
    for split in SPLIT_NAMES:
        for average in ["micro", "macro"]:
            curr = get_acc_df(df, split, average)
            if output is None:
                output = curr
            else:
                output = output.merge(curr, on=exp_specific_columns(output))
    return output


# need to do this to avoid pd hash error
def convert_list_to_tuple(df):
    df.src_domains = df.src_domains.apply(tuple)
    df.target_domains = df.target_domains.apply(tuple)


def assert_acc_rows_are_correct(df):
    # make sure score and split/average columns are equal
    for split in SPLIT_NAMES:
        for average in ["micro", "macro"]:
            curr = get_acc_rows(df, split, average)
            if not curr["score"].equals(curr[acc_score_column_name(split, average)]):
                raise ValueError("These columns should be equal")


# str representation of dict as input
def validator_args_str(validator_args):
    return "_".join([f"{k}_{v}" for k, v in json.loads(validator_args).items()])


def validator_str(validator_name, validator_args):
    v_str = validator_args_str(validator_args)
    return f"{validator_name}_{v_str}"


def domains_str(domains):
    return "_".join(domains)


def task_str(dataset, src_domains, target_domains):
    return f"{dataset}_{domains_str(src_domains)}_{domains_str(target_domains)}"


def add_task_column(df):
    return df.assign(
        task=lambda x: task_str(x["dataset"], x["src_domains"], x["target_domains"])
    )


def unify_validator_columns(df):
    new_col = df.apply(
        lambda x: validator_str(x["validator"], x["validator_args"]), axis=1
    )
    df = df.assign(validator=new_col)
    return df.drop(columns=["validator_args"])


def maybe_per_adapter(df, per_adapter):
    if per_adapter:
        adapters = df["adapter"].unique()
    else:
        adapters = [None]
    return adapters


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
