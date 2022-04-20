import os

import pandas as pd

JOBIDS_FILENAME = "all_validator_jobids.json"
VALIDATOR_TESTS_FOLDER = "validator_tests"
ALL_DFS_FILENAME = "all_dfs.pkl"
PER_SRC_FILENAME = "per_src_threshold.pkl"
PER_SRC_PER_ADAPTER_FILENAME = "per_src_threshold_per_adapter.pkl"
PROCESSED_DF_FILENAME = "all_dfs_processed.pkl"
TARGET_ACCURACY = "target_train_macro"


def get_processed_df(exp_folder):
    filename = os.path.join(exp_folder, PROCESSED_DF_FILENAME)
    if not os.path.isfile(filename):
        return None
    return pd.read_pickle(filename)


def get_per_src_basename(per_adapter, topN, task):
    basename = PER_SRC_PER_ADAPTER_FILENAME if per_adapter else PER_SRC_FILENAME
    return f"{task}_top{topN}_{basename}"


def get_per_src_threshold_df(exp_folder, per_adapter, topN, task):
    basename = get_per_src_basename(per_adapter, topN, task)
    filename = os.path.join(exp_folder, basename)
    if not os.path.isfile(filename):
        return None
    return pd.read_pickle(filename)


def exp_group_args():
    return [
        "exp_groups",
        "exp_group_prefix",
        "exp_group_suffix",
        "exp_group_includes",
        "exp_group_excludes",
    ]


def add_exp_group_args(parser):
    parser.add_argument("--exp_groups", nargs="+", type=str, default=[])
    x = exp_group_args()
    x.remove("exp_groups")
    for k in x:
        parser.add_argument(f"--{k}", type=str)
