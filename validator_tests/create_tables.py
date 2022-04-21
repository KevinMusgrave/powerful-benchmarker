import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from pytorch_adapt.utils import common_functions as c_f

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils
from validator_tests.utils.constants import (
    TARGET_ACCURACY,
    add_exp_group_args,
    get_exp_groups_with_matching_tasks,
    get_name_from_exp_groups,
    get_per_src_threshold_df,
    get_processed_df,
)


def best_accuracy_per_adapter(df, key, exp_groups, tables_folder):
    folder = os.path.join(tables_folder, get_name_from_exp_groups(exp_groups))
    c_f.makedir_if_not_there(folder)
    df = df.groupby(["adapter"])[key].max().reset_index(name=key)
    filename = os.path.join(folder, f"best_accuracy_per_adapter.csv")
    df.to_csv(filename, index=False)


def to_csv(df, folder, key, per_adapter, topN):
    filename = f"{key}"
    if key == "predicted_best_acc":
        filename += f"_top{topN}"
    if per_adapter:
        filename += "_per_adapter"
    filename = os.path.join(folder, f"{filename}.csv")
    df.to_csv(filename, index=False)


def best_validators(df, key, folder, per_adapter, topN):
    c_f.makedir_if_not_there(folder)

    # df = df[(df["validator"] != "Accuracy"]
    group_by = ["validator", "validator_args", key]
    if per_adapter:
        group_by += ["adapter"]
    df = df[[*group_by, "src_threshold"]]
    # If two values match, we want the one with the lowest src threshold
    df = df.groupby(group_by)["src_threshold"].min()
    df = df.reset_index(name="src_threshold")
    df = df.sort_values(by=[key], ascending=False)
    df = df.head(100)
    df.validator_args = df.validator_args.apply(json.loads)
    to_csv(df, folder, key, per_adapter, topN)


def create_best_validators_tables(exp_folder, exp_groups, tables_folder):
    tables_folder = os.path.join(tables_folder, get_name_from_exp_groups(exp_groups))

    for per_adapter in [True, False]:
        topN = args.topN_per_adapter if per_adapter else args.topN
        per_src = get_per_src_threshold_df(exp_folder, per_adapter, topN, exp_groups)
        if per_src is None:
            continue
        best_validators(per_src, "predicted_best_acc", tables_folder, per_adapter, topN)
        # For correlation, only keep src_thresholds where there is a meaningful number of datapoints
        # per_src = per_src[per_src["num_past_threshold"] > 200]
        # print(per_src["num_past_threshold"].max())
        best_validators(per_src, "correlation", tables_folder, per_adapter, topN)


def create_tables(exp_folder, exp_groups, tables_folder, df):
    best_accuracy_per_adapter(df, TARGET_ACCURACY, exp_groups, tables_folder)
    create_best_validators_tables(exp_folder, exp_groups, tables_folder)


def main(args):
    exp_groups = utils.get_exp_groups(args)
    for e in exp_groups:
        print("exp_group", e)
        # do per feature layer
        exp_folder = os.path.join(args.exp_folder, e)
        df = get_processed_df(exp_folder)
        if df is not None:
            create_tables(exp_folder, [e], args.tables_folder, df)

    # # now do with feature layers in one dataframe
    # # which are saved in args.exp_folder
    combined_dfs, combined_exp_groups = get_exp_groups_with_matching_tasks(
        args.exp_folder, exp_groups, return_dfs=True
    )
    for df, e in zip(combined_dfs, combined_exp_groups):
        print("exp_groups", e)
        create_tables(args.exp_folder, e, args.tables_folder, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--tables_folder", type=str, default="tables")

    # topN here refers to the topN used in per_src_threshold.py
    parser.add_argument("--topN", type=int, default=200)
    parser.add_argument("--topN_per_adapter", type=int, default=20)
    args = parser.parse_args()
    main(args)
