import argparse
import sys

import pandas as pd

sys.path.insert(0, ".")
from validator_tests.utils.corr_utils import (
    get_corr_per_task,
    get_corr_per_task_per_adapter,
    get_per_threshold,
)
from validator_tests.utils.df_utils import (
    assert_acc_rows_are_correct,
    convert_list_to_tuple,
    exp_specific_columns,
    get_all_acc,
)
from validator_tests.utils.plot_corr_vs_src import plot_corr_vs_X
from validator_tests.utils.plot_val_vs_acc import plot_val_vs_acc


def main(args):
    df = pd.read_pickle(args.df_path)
    convert_list_to_tuple(df)
    accs = get_all_acc(df)
    df = df.merge(accs, on=exp_specific_columns(df))
    assert_acc_rows_are_correct(df)
    plot_val_vs_acc(df, args.plots_folder)
    per_src, per_target = get_per_threshold(df, get_corr_per_task())
    plot_corr_vs_X("src", False)(per_src, args.plots_folder)
    plot_corr_vs_X("target", False)(per_target, args.plots_folder)

    per_src, per_target = get_per_threshold(df, get_corr_per_task_per_adapter())
    plot_corr_vs_X("src", True)(per_src, args.plots_folder)
    plot_corr_vs_X("target", True)(per_target, args.plots_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--df_path", type=str, required=True)
    parser.add_argument("--plots_folder", type=str, default="plots")
    args = parser.parse_args()
    main(args)
