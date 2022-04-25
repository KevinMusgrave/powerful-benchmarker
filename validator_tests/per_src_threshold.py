import argparse
import os
import sys

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import create_main, utils
from validator_tests.utils.constants import add_exp_group_args
from validator_tests.utils.df_utils import (
    get_exp_groups_with_matching_tasks,
    get_per_src_basename,
    get_processed_df,
)
from validator_tests.utils.threshold_utils import (
    convert_predicted_best_acc_to_rel,
    get_all_per_task_validator,
    get_all_per_task_validator_adapter,
    get_per_threshold,
)


def create_per_x_threshold(df, exp_folder, per_adapter, topN, exp_groups):
    print(f"per_adapter = {per_adapter}")
    fn = (
        get_all_per_task_validator_adapter(topN)
        if per_adapter
        else get_all_per_task_validator(topN)
    )
    per_src = get_per_threshold(df, fn)
    per_src = convert_predicted_best_acc_to_rel(
        df, per_src, per_adapter, topN, len(exp_groups)
    )
    basename = get_per_src_basename(per_src, per_adapter, topN)
    filename = os.path.join(exp_folder, basename)
    print(f"saving to {filename}\n\n")
    per_src.to_pickle(filename)


def run_both(df, exp_folder, topN, topN_per_adapter, exp_groups):
    create_per_x_threshold(df, exp_folder, False, topN, exp_groups)
    create_per_x_threshold(df, exp_folder, True, topN_per_adapter, exp_groups)


def main(args):
    exp_groups = utils.get_exp_groups(args)
    for e in exp_groups:
        exp_folder = os.path.join(args.exp_folder, e)
        df = get_processed_df(exp_folder)
        if df is None:
            continue
        run_both(df, exp_folder, args.topN, args.topN_per_adapter, [e])

    combined_dfs, combined_exp_groups = get_exp_groups_with_matching_tasks(
        args.exp_folder, exp_groups
    )
    for df, e in zip(combined_dfs, combined_exp_groups):
        run_both(df, args.exp_folder, args.topN, args.topN_per_adapter, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    create_main.add_topN_args(parser)
    args = parser.parse_args()
    main(args)
