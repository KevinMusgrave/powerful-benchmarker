import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import derive
from validator_tests.utils.constants import (
    ALL_DFS_FILENAME,
    PER_SRC_FILENAME,
    PER_SRC_PER_ADAPTER_FILENAME,
    PER_TARGET_FILENAME,
    PER_TARGET_PER_ADAPTER_FILENAME,
    PROCESSED_DF_FILENAME,
)
from validator_tests.utils.df_utils import (
    all_acc_score_column_names,
    assert_acc_rows_are_correct,
    convert_list_to_tuple,
    exp_specific_columns,
    get_all_acc,
)
from validator_tests.utils.plot_heat_map import plot_heat_map
from validator_tests.utils.plot_val_vs_acc import plot_val_vs_acc
from validator_tests.utils.plot_vs_threshold import (
    plot_corr_vs_X,
    plot_predicted_best_acc_vs_X,
)
from validator_tests.utils.threshold_utils import (
    get_all_per_task,
    get_all_per_task_per_adapter,
    get_per_threshold,
)


def read_all_dfs(exp_folder):
    df_path = os.path.join(exp_folder, ALL_DFS_FILENAME)
    return pd.read_pickle(df_path)


def process_acc_validator(df):
    accs = get_all_acc(df)
    df = df.merge(accs, on=exp_specific_columns(df, all_acc_score_column_names()))
    assert_acc_rows_are_correct(df)
    return df


def add_derived_scores(df):
    return derive.add_IM(df)


def get_processed_df(exp_folder, read_existing):
    filename = os.path.join(exp_folder, PROCESSED_DF_FILENAME)
    if read_existing and os.path.isfile(filename):
        df = pd.read_pickle(filename)
    else:
        df = read_all_dfs(exp_folder)
        convert_list_to_tuple(df)
        df = add_derived_scores(df)
        df = process_acc_validator(df)
        df.to_pickle(filename)
    return df


def get_per_x_threshold(df, exp_folder, read_existing, per_adapter=False):
    if per_adapter:
        src_basename = PER_SRC_PER_ADAPTER_FILENAME
        target_basename = PER_TARGET_PER_ADAPTER_FILENAME
    else:
        src_basename = PER_SRC_FILENAME
        target_basename = PER_TARGET_FILENAME
    src_filename = os.path.join(exp_folder, src_basename)
    target_filename = os.path.join(exp_folder, target_basename)
    if (
        read_existing
        and os.path.isfile(src_filename)
        and os.path.isfile(target_filename)
    ):
        per_src = pd.read_pickle(src_filename)
        per_target = pd.read_pickle(target_filename)
    else:
        fn = get_all_per_task_per_adapter() if per_adapter else get_all_per_task()
        per_src, per_target = get_per_threshold(df, fn)
        per_src.to_pickle(src_filename)
        per_target.to_pickle(target_filename)
    return per_src, per_target


def main(args):
    exp_folder = os.path.join(args.exp_folder, args.exp_group)
    df = get_processed_df(exp_folder, args.read_existing)
    plot_val_vs_acc(df, args.plots_folder)

    per_src, per_target = get_per_x_threshold(df, exp_folder, args.read_existing)
    plot_corr_vs_X("src", False)(per_src, args.plots_folder)
    plot_corr_vs_X("target", False)(per_target, args.plots_folder)
    plot_predicted_best_acc_vs_X("src", False)(per_src, args.plots_folder)
    plot_predicted_best_acc_vs_X("target", False)(per_target, args.plots_folder)

    per_src, per_target = get_per_x_threshold(
        df, exp_folder, args.read_existing, per_adapter=True
    )
    plot_corr_vs_X("src", True)(per_src, args.plots_folder)
    plot_corr_vs_X("target", True)(per_target, args.plots_folder)
    plot_heat_map(per_src, args.plots_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--exp_group", type=str, required=True)
    parser.add_argument("--plots_folder", type=str, default="plots")
    parser.add_argument("--read_existing", action="store_true")
    args = parser.parse_args()
    main(args)
