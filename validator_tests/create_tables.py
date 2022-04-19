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


def topN_validators(df, key, N, folder, per_adapter=False):
    c_f.makedir_if_not_there(folder)

    df = df[df["validator"] != "Accuracy"]
    group_by = ["validator", "validator_args", key]
    if per_adapter:
        group_by += ["adapter"]
    df = df[[*group_by, "src_threshold"]]
    df = df.groupby(group_by)["src_threshold"].min()
    df = df.reset_index(name="src_threshold")
    df = df.sort_values(by=[key], ascending=False)
    df = df.head(N)
    df.validator_args = df.validator_args.apply(json.loads)

    to_csv(df, folder, key, per_adapter)


def create_tables(args, exp_group):
    exp_folder = os.path.join(args.exp_folder, exp_group)
    df = get_processed_df(exp_folder)
    if df is None:
        return
    feature_layers = df["feature_layer"].unique()
    base_foldername = df["task"].unique()[0]
    if len(feature_layers) == 1:
        base_foldername = f"{base_foldername}_fl{str(feature_layers[0])}"
    tables_folder = os.path.join(args.tables_folder, base_foldername)

    best_accuracy_per_adapter(df, TARGET_ACCURACY, tables_folder)

    # for per_adapter in [True, False]:
    #     per_src = get_per_src_threshold_df(exp_folder, per_adapter)
    #     topN_validators(per_src, "predicted_best_acc", 100, tables_folder, per_adapter)
    #     topN_validators(per_src, "correlation", 100, tables_folder, per_adapter)


def main(args):
    exp_groups = utils.get_exp_groups(args)
    for e in exp_groups:
        create_tables(args, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--tables_folder", type=str, default="tables")
    # parser.add_argument("--topN", type=int, default=100)
    # parser.add_argument("--per_src_topN", type=int, default=200)
    # parser.add_argument("--per_src_per_adapter_topN", type=int, default=20)
    args = parser.parse_args()
    main(args)
