import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils
from validator_tests.utils.constants import (
    add_exp_group_args,
    get_per_src_basename,
    get_processed_df,
)
from validator_tests.utils.threshold_utils import (
    convert_predicted_best_acc_to_rel,
    get_all_per_task_validator,
    get_all_per_task_validator_adapter,
    get_per_threshold,
)


def create_per_x_threshold(df, exp_folder, per_adapter, topN):
    print(f"per_adapter = {per_adapter}")
    tasks = df["task"].unique()
    assert len(tasks) == 1
    print(f"tasks = {tasks}")
    print(f"feature_layers = {df['feature_layer'].unique()}")
    basename = get_per_src_basename(per_adapter, topN, task=tasks[0])
    filename = os.path.join(exp_folder, basename)
    fn = (
        get_all_per_task_validator_adapter(topN)
        if per_adapter
        else get_all_per_task_validator(topN)
    )
    per_src = get_per_threshold(df, fn)
    per_src = convert_predicted_best_acc_to_rel(df, per_src, per_adapter, topN)
    print(f"saving to {filename}\n\n")
    per_src.to_pickle(filename)


def run_both(df, exp_folder, topN, topN_per_adapter):
    create_per_x_threshold(df, exp_folder, False, topN)
    create_per_x_threshold(df, exp_folder, True, topN_per_adapter)


# combined across feature layers
def get_combined_dfs(args, exp_groups):
    num_exp_groups = len(exp_groups)
    combined_dfs = []
    for i in range(num_exp_groups):
        df1 = get_processed_df(os.path.join(args.exp_folder, exp_groups[i]))
        curr_matching = []
        for j in range(i + 1, num_exp_groups):
            df2 = get_processed_df(os.path.join(args.exp_folder, exp_groups[j]))
            if df1 is None or df2 is None:
                continue
            if df1["task"].unique() == df2["task"].unique():
                curr_matching.append(df2)
        if len(curr_matching) > 0:
            combined_dfs.append(pd.concat([df1, *curr_matching], axis=0))
    return combined_dfs


def main(args):
    exp_groups = utils.get_exp_groups(args)
    for e in exp_groups:
        exp_folder = os.path.join(args.exp_folder, e)
        df = get_processed_df(exp_folder)
        if df is None:
            continue
        run_both(df, exp_folder, args.topN, args.topN_per_adapter)

    combined_dfs = get_combined_dfs(args, exp_groups)
    for df in combined_dfs:
        run_both(df, args.exp_folder, args.topN, args.topN_per_adapter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--topN", type=int, default=200)
    parser.add_argument("--topN_per_adapter", type=int, default=20)
    args = parser.parse_args()
    main(args)
