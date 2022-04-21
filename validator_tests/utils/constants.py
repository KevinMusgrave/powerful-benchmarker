import os

import pandas as pd

JOBIDS_FILENAME = "all_validator_jobids.json"
VALIDATOR_TESTS_FOLDER = "validator_tests"
ALL_DFS_FILENAME = "all_dfs.pkl"
PER_SRC_FILENAME = "per_src_threshold.pkl"
PER_SRC_PER_ADAPTER_FILENAME = "per_src_threshold_per_adapter.pkl"
PROCESSED_DF_FILENAME = "all_dfs_processed.pkl"
TARGET_ACCURACY = "target_train_macro"
NUM_ADAPTERS = 10
EXPECTED_NUMBER_OF_CHECKPOINTS = NUM_ADAPTERS * 100 * 20


def get_processed_df(exp_folder):
    filename = os.path.join(exp_folder, PROCESSED_DF_FILENAME)
    if not os.path.isfile(filename):
        return None
    return pd.read_pickle(filename)


def get_name_from_exp_groups(exp_groups):
    split_names = [i.split("_") for i in exp_groups]
    return "_".join(["".join(sorted(list(set(i)))) for i in zip(*split_names)])


def get_per_src_basename(per_adapter, topN, exp_groups):
    basename = PER_SRC_PER_ADAPTER_FILENAME if per_adapter else PER_SRC_FILENAME
    exp_groups = get_name_from_exp_groups(exp_groups)
    return f"{exp_groups}_top{topN}_{basename}"


def get_per_src_threshold_df(exp_folder, per_adapter, topN, exp_groups):
    basename = get_per_src_basename(per_adapter, topN, exp_groups)
    filename = os.path.join(exp_folder, basename)
    if not os.path.isfile(filename):
        return None
    return pd.read_pickle(filename)


def tasks_match(e1, e2):
    return e1.split("_fl")[0] == e2.split("_fl")[0]


# combined across feature layers etc
def get_exp_groups_with_matching_tasks(exp_folder, exp_groups, return_dfs=False):
    num_exp_groups = len(exp_groups)
    combined_dfs, combined_exp_groups = [], []
    for i in range(num_exp_groups):
        e1 = exp_groups[i]
        if any(e1 in ceg for ceg in combined_exp_groups):
            continue

        curr_exp_groups = []
        if return_dfs:
            df1 = get_processed_df(os.path.join(exp_folder, e1))
            curr_dfs = []

        for j in range(i + 1, num_exp_groups):
            e2 = exp_groups[j]
            if not tasks_match(e1, e2):
                continue
            if return_dfs:
                df2 = get_processed_df(os.path.join(exp_folder, e2))
                if df1 is None or df2 is None:
                    continue
                assert df1["task"].unique() == df2["task"].unique()
                curr_dfs.append(df2)
            curr_exp_groups.append(e2)

        if len(curr_exp_groups) > 0:
            if return_dfs:
                combined_dfs.append(pd.concat([df1, *curr_dfs], axis=0))
            combined_exp_groups.append((e1, *curr_exp_groups))

    if return_dfs:
        return combined_dfs, combined_exp_groups
    return combined_exp_groups


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
