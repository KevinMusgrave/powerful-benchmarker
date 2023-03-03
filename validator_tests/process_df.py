import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import derive, utils
from validator_tests.utils.constants import PROCESSED_DF_FILENAME, add_exp_group_args
from validator_tests.utils.df_utils import (
    add_task_column,
    all_acc_score_column_names,
    assert_acc_rows_are_correct,
    convert_list_to_tuple,
    drop_irrelevant_columns,
    exp_specific_columns,
    get_all_acc,
    get_all_dfs,
)


def unused_bnm_args():
    return [
        '{"layer": "features", "split": "src_train"}',
        '{"layer": "features", "split": "src_val"}',
        '{"layer": "features", "split": "target_train"}',
        '{"layer": "preds", "split": "src_train"}',
        '{"layer": "preds", "split": "src_val"}',
        '{"layer": "preds", "split": "target_train"}',
    ]


def unused_mmdperclass_args():
    return [
        '{"exponent": 0, "layer": "preds", "normalize": true, "split": "train"}',
        '{"exponent": 0, "layer": "preds", "normalize": false, "split": "train"}',
    ]


def expected_num_validators():
    accuracy = 8
    entropy = 3
    # diversity = 3
    dev_binary = 9
    snd = 12
    class_ami = 8
    class_ss = 8
    # mmd = 6
    # mmd_per_class = 4
    bnm = 3
    return (
        accuracy
        + entropy
        # + diversity
        + dev_binary
        + snd
        + class_ami
        + class_ss
        # + mmd
        # + mmd_per_class
        + bnm
    )


def filter_validators(df):
    df = df[
        df["validator"].isin(
            [
                "Accuracy",
                "Entropy",
                # "Diversity",
                "DEVBinary",
                "SND",
                "ClassAMICentroidInit",
                "ClassSSCentroidInit",
                # "MMD",
                # "MMDPerClass",
                "BNM",
                # "MCC",
                # "NearestSource",
                # "NearestSourceL2",
            ]
        )
    ]

    df = df[
        ~((df["validator"] == "BNM") & (df["validator_args"].isin(unused_bnm_args())))
    ]

    df = df[
        ~(
            (df["validator"] == "MMDPerClass")
            & (df["validator_args"].isin(unused_mmdperclass_args()))
        )
    ]

    return df


def keep_common_experiments(df):
    groupby = ["dataset", "adapter", "exp_name", "trial_num", "trial_params", "epoch"]
    size_per_exp = df.groupby(groupby).size()
    size_per_exp = size_per_exp[size_per_exp == expected_num_validators()].reset_index()
    keep = True
    for g in groupby:
        keep &= df[g].isin(size_per_exp[g])
    return df[keep]


def process_acc_validator(df):
    accs = get_all_acc(df)
    df = df.merge(accs, on=exp_specific_columns(df, all_acc_score_column_names()))
    assert_acc_rows_are_correct(df)
    return df


def assert_all_same_size(df):
    size_per_validator = df.groupby(["validator", "validator_args"]).size()
    unique_sizes = size_per_validator.unique()
    if len(unique_sizes) != 1:
        print("error: all validators should have same number of elements")
        for x in unique_sizes:
            print(f"\nvalidators with size={x}")
            print(size_per_validator[size_per_validator == x])
        raise ValueError


def copy_epoch_0_rows(df):
    new_rows = []
    num_trials = 0
    epoch_0_rows = df[df["exp_name"] == "epoch_0"]
    for exp_name in df["exp_name"].unique():
        if exp_name == "epoch_0":
            continue
        curr_df = df[df["exp_name"] == exp_name]
        adapter = curr_df["adapter"].unique().item()
        dataset = curr_df["dataset"].unique().item()
        src_domains = curr_df["src_domains"].unique().item()
        target_domains = curr_df["target_domains"].unique().item()
        for trial_num in curr_df["trial_num"].unique():
            num_trials += 1
            curr_0_rows = epoch_0_rows.copy()
            curr_0_rows["adapter"] = adapter
            curr_0_rows["exp_name"] = exp_name
            curr_0_rows["trial_num"] = trial_num
            assert len(curr_0_rows) == expected_num_validators()
            assert curr_0_rows.dataset.unique().item() == dataset
            assert curr_0_rows.src_domains.unique().item() == src_domains
            assert curr_0_rows.target_domains.unique().item() == target_domains
            new_rows.append(curr_0_rows)

    new_df = pd.concat([df, *new_rows], axis=0)
    new_df = new_df[new_df["exp_name"] != "epoch_0"]
    new_len = len(new_df)
    correct_len = (
        len(df)
        + (expected_num_validators() * num_trials)
        - len(df[df["exp_name"] == "epoch_0"])
    )
    assert new_len == correct_len

    for adapter in df["adapter"].unique():
        if adapter == "PretrainerConfig":
            continue
        old_df = df[df["adapter"] == adapter]
        num_trials = len(old_df["trial_num"].unique())
        print("num_trials", num_trials)
        new_len = len(new_df[new_df["adapter"] == adapter])
        expected_len = len(old_df) + (expected_num_validators() * num_trials)
        print(adapter, new_len, expected_len)
        assert new_len == expected_len

    return new_df


def process_df(args, exp_group):
    exp_folder = os.path.join(args.exp_folder, exp_group)
    filename = os.path.join(exp_folder, PROCESSED_DF_FILENAME)

    print("reading file")
    df = get_all_dfs(exp_folder)
    if df is None:
        return

    print("drop_irrelevant_columns")
    df = drop_irrelevant_columns(df)

    print("convert_list_to_tuple")
    convert_list_to_tuple(df)

    print("filtering validators")
    df = filter_validators(df)

    print("copying epoch_0 rows")
    df = copy_epoch_0_rows(df)

    print("keep common experiments")
    df = keep_common_experiments(df)

    print("processing accuracies")
    df = process_acc_validator(df)
    if len(df) == 0:
        print("accuracies have not been computed yet. Exiting")
        return

    print("add_task_column")
    df = add_task_column(df)

    print("adding derived scores")
    df = derive.add_derived_scores(df)
    assert_all_same_size(df)

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
    args = parser.parse_args()
    main(args)
