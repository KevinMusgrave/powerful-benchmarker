import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from pytorch_adapt.utils import common_functions as c_f

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils.constants import get_per_src_threshold_df, get_processed_df
from validator_tests.utils.threshold_utils import TARGET_ACCURACY


def to_csv(df, folder, key, per_adapter=False):
    filename = f"top_{key}"
    if per_adapter:
        filename += "_per_adapter"
    filename = os.path.join(folder, f"{filename}.csv")
    df.to_csv(filename, index=False)


def topN_accuracies(df, key, N, folder):
    c_f.makedir_if_not_there(folder)
    df = df.drop(columns=["score", "validator", "validator_args"])
    df = df.drop_duplicates(ignore_index=True)
    df = df[["adapter", key]]
    df = df.sort_values(by=[key], ascending=False)
    df = df.head(N)
    to_csv(df, folder, key)


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


def main(args):
    exp_folder = os.path.join(args.exp_folder, args.exp_group)
    tables_folder = args.tables_folder

    df = get_processed_df(exp_folder)
    topN_accuracies(df, TARGET_ACCURACY, 100, tables_folder)

    for per_adapter in [True, False]:
        per_src = get_per_src_threshold_df(exp_folder, per_adapter)
        topN_validators(per_src, "predicted_best_acc", 100, tables_folder, per_adapter)
        topN_validators(per_src, "correlation", 100, tables_folder, per_adapter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--exp_group", type=str, required=True)
    parser.add_argument("--tables_folder", type=str, default="tables")
    args = parser.parse_args()
    main(args)
