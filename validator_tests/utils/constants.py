import os

import pandas as pd

JOBIDS_FILENAME = "all_validator_jobids.json"
VALIDATOR_TESTS_FOLDER = "validator_tests"
ALL_DFS_FILENAME = "all_dfs.pkl"
PER_SRC_FILENAME = "per_src_threshold.pkl"
PER_TARGET_FILENAME = "per_target_threshold.pkl"
PER_SRC_PER_ADAPTER_FILENAME = "per_src_threshold_per_adapter.pkl"
PER_TARGET_PER_ADAPTER_FILENAME = "per_target_threshold_per_adapter.pkl"
PER_TARGET_FILENAME = "per_target_threshold.pkl"
PROCESSED_DF_FILENAME = "all_dfs_processed.pkl"


def get_processed_df(exp_folder):
    filename = os.path.join(exp_folder, PROCESSED_DF_FILENAME)
    return pd.read_pickle(filename)


def get_per_src_basename(per_adapter):
    return PER_SRC_PER_ADAPTER_FILENAME if per_adapter else PER_SRC_FILENAME


def get_per_src_threshold_df(exp_folder, per_adapter):
    basename = get_per_src_basename(per_adapter)
    filename = os.path.join(exp_folder, basename)
    return pd.read_pickle(filename)


def add_exp_group_args(parser):
    parser.add_argument("--exp_groups", nargs="+", type=str, default=[])
    for x in ["exp_group_prefix", "exp_group_suffix", "exp_group_contains"]:
        parser.add_argument(f"--{x}", type=str)
