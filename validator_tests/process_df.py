import argparse
import json
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
    unify_validator_columns,
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


def process_acc_validator(df):
    sizes_before = df.groupby(["validator", "validator_args"]).size()
    accs = get_all_acc(df)
    df = df.merge(accs, on=exp_specific_columns(df, all_acc_score_column_names()))
    assert_acc_rows_are_correct(df)
    sizes_after = df.groupby(["validator", "validator_args"]).size()
    mismatch_mask = sizes_before != sizes_after
    greater_than_mask = sizes_after > sizes_before
    if np.sum(mismatch_mask) > 0:
        print("WARNING: sizes before and after don't match in process_acc_validator")
        # raise ValueError(f"sizes before and after don't match\n\n{sizes_before[mismatch_mask]}\n\n{sizes_after[mismatch_mask]}")
    if np.sum(greater_than_mask) > 0:
        raise ValueError("sizes_after has values greater than sizes_before")
    return df


def warn_unfinished_validators(df, detailed_warnings):
    df = unify_validator_columns(df)
    unfinished = {}
    too_many = {}
    for v in df["validator"].unique():
        num_done = len(df[df["validator"] == v])
        if num_done < EXPECTED_NUMBER_OF_CHECKPOINTS:
            unfinished[v] = num_done
        if num_done > EXPECTED_NUMBER_OF_CHECKPOINTS:
            too_many[v] = num_done
    if len(unfinished) > 0:
        print("WARNING: the following validators haven't finished")
        if detailed_warnings:
            print(json.dumps(unfinished, indent=4, sort_keys=True))
    if len(too_many) > 0:
        print("WARNING: the following validators have more entries than expected")
        if detailed_warnings:
            print(json.dumps(too_many, indent=4, sort_keys=True))


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
    df = process_acc_validator(df)
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
    print_validators_with_nan(df)
    df = remove_nan_inf_scores(df)
    print_validators_with_nan(df, assert_none=True)

    print(f"processed df:\n{df}")
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
