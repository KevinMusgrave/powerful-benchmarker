import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import derive, utils
from validator_tests.utils.constants import (
    EXPECTED_NUMBER_OF_CHECKPOINTS,
    PROCESSED_DF_FILENAME,
    add_exp_group_args,
)
from validator_tests.utils.df_utils import (
    add_task_column,
    all_acc_score_column_names,
    assert_acc_rows_are_correct,
    convert_list_to_tuple,
    drop_irrelevant_columns,
    exp_specific_columns,
    get_all_acc,
    get_all_dfs,
    print_validators_with_nan,
    remove_nan_inf_scores,
)


def filter_validators(df):
    return df[
        df["validator"].isin(
            [
                "Accuracy",
                "Entropy",
                "Diversity",
                "DEVBinary",
                "SND",
                "ClassAMICentroidInit",
                "MMD",
                "MMDPerClass",
                "BNM",
            ]
        )
    ]


def process_acc_validator(df, detailed_warnings):
    v_keys = ["validator", "validator_args"]
    sizes_before = df.groupby(v_keys).size().reset_index(name="before")
    accs = get_all_acc(df)
    df = df.merge(accs, on=exp_specific_columns(df, all_acc_score_column_names()))
    assert_acc_rows_are_correct(df)
    sizes_after = df.groupby(v_keys).size().reset_index(name="after")
    sizes_before = sizes_before.merge(sizes_after, on=v_keys)
    mismatch_mask = sizes_before["before"] != sizes_before["after"]
    greater_than_mask = sizes_before["before"] < sizes_before["after"]
    if np.sum(mismatch_mask) > 0:
        print("WARNING: sizes before and after don't match in process_acc_validator")
        if detailed_warnings:
            print(sizes_before[mismatch_mask])
    if np.sum(greater_than_mask) > 0:
        raise ValueError("sizes after has values greater than sizes before")
    return df


def warn_unfinished_validators(df, detailed_warnings):
    num_done = df.groupby(["validator", "validator_args"]).size()
    num_done = num_done[num_done != EXPECTED_NUMBER_OF_CHECKPOINTS]
    if len(num_done) > 0:
        print(
            f"WARNING: there are {len(num_done)} validators with less/more entries than expected"
        )
        if detailed_warnings:
            print(num_done)


def process_df(args, exp_group):
    exp_folder = os.path.join(args.exp_folder, exp_group)
    filename = os.path.join(exp_folder, PROCESSED_DF_FILENAME)

    print("reading file")
    df = get_all_dfs(exp_folder)
    if df is None:
        return
    print("convert_list_to_tuple")
    convert_list_to_tuple(df)

    print("filtering validators")
    df = filter_validators(df)

    print("processing accuracies")
    df = process_acc_validator(df, args.detailed_warnings)
    if len(df) == 0:
        print("accuracies have not been computed yet. Exiting")
        return

    print("drop_irrelevant_columns")
    df = drop_irrelevant_columns(df)

    print("add_task_column")
    df = add_task_column(df)

    print("adding derived scores")
    df = derive.add_derived_scores(df)

    print("finding unfinished validators")
    warn_unfinished_validators(df, args.detailed_warnings)
    df = remove_nan_inf_scores(df)
    print_validators_with_nan(df, assert_none=True)

    print(f"saving df to {filename}")
    df.to_pickle(filename)


def main(args):
    exp_groups = utils.get_exp_groups(args)
    for e in exp_groups:
        process_df(args, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--detailed_warnings", action="store_true")
    args = parser.parse_args()
    main(args)
