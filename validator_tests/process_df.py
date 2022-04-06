import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import derive
from validator_tests.utils.constants import ALL_DFS_FILENAME, PROCESSED_DF_FILENAME
from validator_tests.utils.df_utils import (
    all_acc_score_column_names,
    assert_acc_rows_are_correct,
    convert_list_to_tuple,
    exp_specific_columns,
    get_all_acc,
    print_validators_with_nan,
    remove_nan_scores,
    unify_validator_columns,
)

EXPECTED_NUMBER_OF_CHECKPOINTS = 10 * 100 * 20


def read_all_dfs(exp_folder):
    df_path = os.path.join(exp_folder, ALL_DFS_FILENAME)
    return pd.read_pickle(df_path)


def process_acc_validator(df):
    accs = get_all_acc(df)
    df = df.merge(accs, on=exp_specific_columns(df, all_acc_score_column_names()))
    assert_acc_rows_are_correct(df)
    return df


def warn_unfinished_validators(df):
    df = unify_validator_columns(df)
    unfinished = {}
    for v in df["validator"].unique():
        num_done = len(df[df["validator"] == v])
        if num_done < EXPECTED_NUMBER_OF_CHECKPOINTS:
            unfinished[v] = num_done
    if len(unfinished) > 0:
        print("WARNING: the following validators haven't finished")
        print(json.dumps(unfinished, indent=4, sort_keys=True))


def main(args):
    exp_folder = os.path.join(args.exp_folder, args.exp_group)
    filename = os.path.join(exp_folder, PROCESSED_DF_FILENAME)

    print("reading file")
    df = read_all_dfs(exp_folder)
    convert_list_to_tuple(df)

    print("adding derived scores")
    df = derive.add_derived_scores(df)
    print("finding unfinished validators")
    warn_unfinished_validators(df)

    print("processing accuracies")
    df = process_acc_validator(df)
    print_validators_with_nan(df)
    df = remove_nan_scores(df)
    print_validators_with_nan(df, assert_none=True)

    print(f"processed df:\n{df}")
    df.to_pickle(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--exp_group", type=str, required=True)
    args = parser.parse_args()
    main(args)
