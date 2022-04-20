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
    get_per_src_threshold_df,
    get_processed_df,
)


def best_accuracy_per_adapter(df, key, folder):
    c_f.makedir_if_not_there(folder)
    df = df.groupby(["adapter"])[key].max().reset_index(name=key)
    filename = os.path.join(folder, f"best_accuracy_per_adapter.csv")
    df.to_csv(filename, index=False)


def to_csv(df, folder, key, per_adapter=False):
    filename = f"top_{key}"
    if per_adapter:
        filename += "_per_adapter"
    filename = os.path.join(folder, f"{filename}.csv")
    df.to_csv(filename, index=False)


def best_validators(df, key, folder, per_adapter=False):
    c_f.makedir_if_not_there(folder)

    df = df[df["validator"] != "Accuracy"]
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
    to_csv(df, folder, key, per_adapter)


def get_tables_folder(args, task, feature_layers):
    if len(feature_layers) == 1:
        base_foldername = f"{task}_fl{str(feature_layers[0])}"
    else:
        base_foldername = task
    return os.path.join(args.tables_folder, base_foldername)


def create_best_validators_tables(per_src, tables_folder, per_adapter):
    best_validators(per_src, "predicted_best_acc", tables_folder, per_adapter)
    # For correlation, only keep src_thresholds where there is a meaningful number of datapoints
    # per_src = per_src[per_src["num_past_threshold"] > 200]
    # print(per_src["num_past_threshold"].max())
    best_validators(per_src, "correlation", tables_folder, per_adapter)


def create_tables(args, exp_group):
    print("exp_group", exp_group)
    exp_folder = os.path.join(args.exp_folder, exp_group)
    df = get_processed_df(exp_folder)
    if df is None:
        return
    task = df["task"].unique()[0]
    feature_layers = df["feature_layer"].unique()
    tables_folder = get_tables_folder(args, task, feature_layers)
    best_accuracy_per_adapter(df, TARGET_ACCURACY, tables_folder)

    for per_adapter in [True, False]:
        topN = args.topN_per_adapter if per_adapter else args.topN
        per_src = get_per_src_threshold_df(exp_folder, per_adapter, topN, task)
        create_best_validators_tables(per_src, tables_folder, per_adapter)

    return task


def main(args):
    exp_groups = utils.get_exp_groups(args)
    for e in exp_groups:
        # do per feature layer
        task = create_tables(args, e)
        if task is None:
            continue
        # now do with feature layers in one dataframe
        # which are saved in args.exp_folder
        for per_adapter in [True, False]:
            topN = args.topN_per_adapter if per_adapter else args.topN
            per_src = get_per_src_threshold_df(args.exp_folder, per_adapter, topN, task)
            if per_src is None:
                continue
            tables_folder = get_tables_folder(args, task, [])
            create_best_validators_tables(per_src, tables_folder, per_adapter)


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
