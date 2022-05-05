import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from pytorch_adapt.utils import common_functions as c_f

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import create_main
from validator_tests.utils.constants import TARGET_ACCURACY, add_exp_group_args
from validator_tests.utils.df_utils import get_name_from_df, get_per_src_threshold_df


def best_accuracy_per_adapter(df, key, tables_folder):
    folder = os.path.join(tables_folder, get_name_from_df(df, assert_one_task=True))
    c_f.makedir_if_not_there(folder)
    df = df.groupby(["adapter", "task"])[key].max().reset_index(name=key)
    filename = os.path.join(folder, f"best_accuracy_per_adapter")
    df.to_csv(f"{filename}.csv", index=False)
    df.to_pickle(f"{filename}.pkl")


def to_csv_and_pickle(df, folder, key, per_adapter, topN, src_threshold):
    filename = f"{key}"
    if key == "predicted_best_acc":
        filename += f"_top{topN}"
    if per_adapter:
        filename += "_per_adapter"
    filename += f"_{src_threshold}_src_threshold"
    filename = os.path.join(folder, f"{filename}")
    df.to_csv(f"{filename}.csv", index=False)
    df.to_pickle(f"{filename}.pkl")


def best_validators(df, key, folder, per_adapter, topN, src_threshold):
    c_f.makedir_if_not_there(folder)

    group_by = ["validator", "validator_args", "task"]
    if per_adapter:
        group_by += ["adapter"]
    df = df[df["src_threshold"] == src_threshold]
    min_num_past_threshold = df["num_past_threshold"].min()
    if min_num_past_threshold < topN:
        raise ValueError(f"{min_num_past_threshold} < {topN}")

    df = df.groupby([*group_by])[key].max().reset_index(name=key)
    df = df.sort_values(by=[key], ascending=False)
    to_csv_and_pickle(df, folder, key, per_adapter, topN, src_threshold)


def create_best_validators_tables(exp_folder, exp_groups, tables_folder):
    for per_adapter in [False]:
        topN = args.topN_per_adapter if per_adapter else args.topN
        per_src = get_per_src_threshold_df(exp_folder, per_adapter, topN, exp_groups)
        if per_src is None:
            continue
        curr_folder = os.path.join(
            tables_folder, get_name_from_df(per_src, assert_one_task=True)
        )
        for src_threshold in [0, 0.9]:
            best_validators(
                per_src,
                "predicted_best_acc",
                curr_folder,
                per_adapter,
                topN,
                src_threshold,
            )
            best_validators(
                per_src, "correlation", curr_folder, per_adapter, topN, src_threshold
            )


def create_tables(exp_folder, exp_groups, tables_folder, df):
    best_accuracy_per_adapter(df, TARGET_ACCURACY, tables_folder)
    create_best_validators_tables(exp_folder, exp_groups, tables_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    # topN here refers to the topN used in per_src_threshold.py
    create_main.add_topN_args(parser)
    parser.add_argument("--output_folder", type=str, default="tables")
    args = parser.parse_args()
    create_main.main(args, create_tables, create_tables)
