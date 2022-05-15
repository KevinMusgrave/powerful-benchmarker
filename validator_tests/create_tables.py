import argparse
import os
import sys

sys.path.insert(0, ".")
from pytorch_adapt.utils import common_functions as c_f

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import create_main
from validator_tests.utils.constants import (
    TARGET_ACCURACY,
    TARGET_VAL_ACCURACY,
    add_exp_group_args,
)
from validator_tests.utils.df_utils import (
    get_acc_rows,
    get_name_from_df,
    get_per_src_threshold_df,
)


def best_accuracy_per_adapter(df, tables_folder):
    folder = os.path.join(tables_folder, get_name_from_df(df, assert_one_task=True))
    c_f.makedir_if_not_there(folder)
    for suffix in ["", "_val"]:
        split = TARGET_ACCURACY if suffix == "" else TARGET_VAL_ACCURACY
        curr_df = df.groupby(["adapter", "task"], as_index=False)[split].max()
        curr_df = curr_df[["adapter", "task", split]]
        filename = os.path.join(folder, f"best_accuracy_per_adapter{suffix}")
        curr_df.to_csv(f"{filename}.csv", index=False)
        curr_df.to_pickle(f"{filename}.pkl")


def best_accuracy_topN(df, folder, per_adapter, topN):
    df = df[df["src_threshold"] == 0]
    for suffix in ["", "_val"]:
        split = TARGET_ACCURACY if suffix == "" else TARGET_VAL_ACCURACY
        best_str = f"best_acc{suffix}"
        best_std_str = f"{best_str}_std"
        to_save = ["task", best_str, best_std_str]
        if per_adapter:
            to_save = ["adapter"] + to_save
        curr_df = df[to_save].drop_duplicates()
        curr_df = curr_df.rename(
            columns={best_str: split, best_std_str: f"{split}_std"}
        )
        to_csv_and_pickle(curr_df, folder, f"best_accuracy{suffix}", per_adapter, topN)


def to_csv_and_pickle(df, folder, key, per_adapter, topN, src_threshold=None):
    filename = f"{key}"
    if any(
        key.startswith(k)
        for k in [
            "predicted_best_acc",
            "highest_src_threshold_possible",
            "best_accuracy",
        ]
    ):
        filename += f"_top{topN}"
    if per_adapter:
        filename += "_per_adapter"
    if src_threshold is not None:
        filename += f"_{src_threshold}_src_threshold"
    filename = os.path.join(folder, f"{filename}")
    df.to_csv(f"{filename}.csv", index=False)
    df.to_pickle(f"{filename}.pkl")


def get_group_by(per_adapter):
    group_by = ["validator", "validator_args", "task"]
    if per_adapter:
        group_by += ["adapter"]
    return group_by


def ignore_num_past_threshold_less_than_topN(df, key, topN):
    # ignore rows where num_past_threshold is less than topN
    df.loc[df["num_past_threshold"] < topN, key] = float("nan")


def best_validators(df, key, folder, per_adapter, topN, src_threshold):
    c_f.makedir_if_not_there(folder)
    is_predicted_best_acc = key.startswith("predicted_best_acc")
    group_by = get_group_by(per_adapter)
    df = df[df["src_threshold"] == src_threshold]
    if is_predicted_best_acc:
        ignore_num_past_threshold_less_than_topN(df, key, topN)
    df = df.loc[df.groupby(group_by)[key].idxmax().dropna()]
    df = df.sort_values(by=[key], ascending=False)
    columns = group_by + [key]
    if is_predicted_best_acc:
        columns += [f"{key}_std"]
    df = df[columns]
    if per_adapter:
        group_by.remove("adapter")
        df = df.pivot(index=group_by, columns="adapter")
        if is_predicted_best_acc:
            df.columns.names = (None, None)
        else:
            df = df.droplevel(0, axis=1).rename_axis(None, axis=1).reset_index()
    to_csv_and_pickle(df, folder, key, per_adapter, topN, src_threshold)


def highest_src_threshold_possible(df, folder, per_adapter, topN):
    df = get_acc_rows(df, "target_train", "micro")
    ignore_num_past_threshold_less_than_topN(df, "predicted_best_acc", topN)
    group_by = get_group_by(per_adapter) + ["predicted_best_acc"]
    df = df.groupby(group_by, as_index=False)["src_threshold"].max()
    df = df[df["predicted_best_acc"] == 1]
    to_save = ["task", "predicted_best_acc", "src_threshold"]
    if per_adapter:
        to_save = ["adapter"] + to_save
    df = df[to_save]
    to_csv_and_pickle(
        df, folder, "highest_src_threshold_possible", per_adapter, topN, None
    )


def create_best_validators_tables(exp_folder, exp_groups, tables_folder):
    for per_adapter in [False, True]:
        topN = args.topN_per_adapter if per_adapter else args.topN
        per_src = get_per_src_threshold_df(exp_folder, per_adapter, topN, exp_groups)
        if per_src is None:
            continue
        curr_folder = os.path.join(
            tables_folder, get_name_from_df(per_src, assert_one_task=True)
        )
        best_accuracy_topN(per_src.copy(), curr_folder, per_adapter, topN)
        highest_src_threshold_possible(per_src.copy(), curr_folder, per_adapter, topN)
        for src_threshold in [0, 0.87]:
            for suffix in ["", "_val"]:
                best_validators(
                    per_src.copy(),
                    f"predicted_best_acc{suffix}",
                    curr_folder,
                    per_adapter,
                    topN,
                    src_threshold,
                )
                best_validators(
                    per_src.copy(),
                    f"correlation{suffix}",
                    curr_folder,
                    per_adapter,
                    topN,
                    src_threshold,
                )


def create_tables(exp_folder, exp_groups, tables_folder, df):
    best_accuracy_per_adapter(df, tables_folder)
    create_best_validators_tables(exp_folder, exp_groups, tables_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    # topN here refers to the topN used in per_src_threshold.py
    create_main.add_main_args(parser)
    parser.add_argument("--output_folder", type=str, default="tables")
    args = parser.parse_args()
    create_main.main(args, create_tables, create_tables)
