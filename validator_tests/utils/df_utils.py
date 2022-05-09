import json
import os

import numpy as np
import pandas as pd

from powerful_benchmarker.utils.utils import create_exp_group_name

from .constants import (
    ALL_DFS_FILENAME,
    PER_SRC_FILENAME,
    PER_SRC_PER_ADAPTER_FILENAME,
    PROCESSED_DF_FILENAME,
)
from .utils import dict_to_str, validator_str

SPLIT_NAMES = ["src_train", "src_val", "target_train", "target_val"]
AVERAGE_NAMES = ["micro", "macro"]


def exp_specific_columns(df, additional_exclude=None, exclude=None):
    if exclude is None:
        exclude = ["score", "validator", "validator_args"]
        if additional_exclude:
            exclude.extend(additional_exclude)
    return [x for x in df.columns.values if x not in exclude]


def acc_score_column_name(split, average):
    return f"{split}_{average}"


def all_acc_score_column_names():
    return [acc_score_column_name(x, y) for x in SPLIT_NAMES for y in AVERAGE_NAMES]


def get_acc_rows(df, split, average):
    args = dict_to_str({"average": average, "split": split})
    return df[(df["validator_args"] == args) & (df["validator"] == "Accuracy")]


def drop_validator_cols(df, drop_validator_args=True):
    cols = ["validator"]
    if drop_validator_args:
        cols.append("validator_args")
    return df.drop(columns=cols)


def get_acc_df(df, split, average):
    df = get_acc_rows(df, split, average)
    df = drop_validator_cols(df)
    return df.rename(columns={"score": acc_score_column_name(split, average)})


def get_all_acc(df):
    output = None
    for split in SPLIT_NAMES:
        for average in ["micro", "macro"]:
            curr = get_acc_df(df, split, average)
            if output is None:
                output = curr
            else:
                output = output.merge(
                    curr, on=exp_specific_columns(output, all_acc_score_column_names())
                )
    return output


# need to do this to avoid pd hash error
def convert_list_to_tuple(df):
    df.src_domains = df.src_domains.apply(tuple)
    df.target_domains = df.target_domains.apply(tuple)


def assert_acc_rows_are_correct(df):
    # make sure score and split/average columns are equal
    for split in SPLIT_NAMES:
        for average in ["micro", "macro"]:
            curr = get_acc_rows(df, split, average)
            if not curr["score"].equals(curr[acc_score_column_name(split, average)]):
                raise ValueError("These columns should be equal")


def drop_irrelevant_columns(df):
    return df.drop(
        columns=[
            "exp_folder",
            "dataset_folder",
            "num_workers",
            "evaluate",
            "save_features",
            "download_datasets",
            "use_stat_getter",
            "check_initial_score",
            "use_full_inference",
            "exp_validator",
        ]
    )


def domains_str(domains):
    return "_".join(domains)


def task_str(dataset, src_domains, target_domains):
    return f"{dataset}_{domains_str(src_domains)}_{domains_str(target_domains)}"


def add_task_column(df):
    new_col = df.apply(
        lambda x: task_str(x["dataset"], x["src_domains"], x["target_domains"]), axis=1
    )
    return df.assign(task=new_col)


def task_name_split(name):
    # returns dataset, src domain, target domain
    # TODO support multiple src domains and target domains
    return name.split("_")


def accuracy_name_split(name):
    # returns src/target, train/val, micro/macro
    return name.split("_")


def unify_validator_columns(df):
    new_col = df.apply(
        lambda x: validator_str(x["validator"], x["validator_args"]), axis=1
    )
    df = df.assign(validator=new_col)
    return df.drop(columns=["validator_args"])


def maybe_per_adapter(df, per_adapter):
    if per_adapter:
        adapters = df["adapter"].unique()
    else:
        adapters = [None]
    return adapters


def print_validators_with_nan(df, return_df=False, assert_none=False):
    for fn in [np.isnan, np.isinf]:
        curr_df = df[fn(df["score"])]
        if return_df:
            return curr_df
        curr_df = curr_df[["validator", "validator_args"]].drop_duplicates()
        if len(curr_df) > 0:
            type_str = "NaN" if fn is np.isnan else "inf"
            print(f"WARNING: the following validator/validator_args have {type_str}")
            print(curr_df)
            if assert_none:
                raise ValueError("There should be no scores with nan or inf")


def remove_nan_inf_scores(df):
    mask = np.isnan(df["score"]) | np.isinf(df["score"])
    return df[~mask]


def remove_arg(df, to_remove):
    x = json.loads(df["validator_args"])
    return dict_to_str({k: v for k, v in x.items() if k not in to_remove})


def remove_arg_from_validator_args(df, to_remove):
    new_col = df.apply(lambda y: remove_arg(y, to_remove), axis=1)
    return df.assign(validator_args=new_col)


def get_sorted_unique(df, key, assert_one=False):
    x = df[key].unique()
    if assert_one and len(x) != 1:
        raise ValueError(f"There should be only 1 {key}")
    return tuple(sorted(x))


def unique_tuples_to_sorted_list(df, key):
    output = sorted(df[key].unique())
    if isinstance(output[0], tuple):
        output = [i for sub in output for i in sub]
        output = sorted(list(set(output)))
    return output


def get_name_from_df(df, assert_one_task=False):
    return create_exp_group_name(
        dataset=get_sorted_unique(df, "dataset", assert_one=assert_one_task)[0],
        src_domains=get_sorted_unique(df, "src_domains", assert_one=assert_one_task)[0],
        target_domains=get_sorted_unique(
            df, "target_domains", assert_one=assert_one_task
        )[0],
        feature_layers=unique_tuples_to_sorted_list(df, "feature_layer"),
        optimizers=unique_tuples_to_sorted_list(df, "optimizer"),
        lr_multipliers=unique_tuples_to_sorted_list(df, "lr_multiplier"),
    )


def get_name_from_exp_groups(exp_groups):
    split_names = [i.split("_") for i in exp_groups]
    return "_".join("".join(sorted(list(set(i)))) for i in zip(*split_names))


def get_per_src_basename(per_adapter, topN, df=None, exp_groups=None):
    basename = PER_SRC_PER_ADAPTER_FILENAME if per_adapter else PER_SRC_FILENAME
    exp_group = (
        get_name_from_df(df) if df is not None else get_name_from_exp_groups(exp_groups)
    )
    return f"{exp_group}_top{topN}_{basename}"


def get_per_src_threshold_df(exp_folder, per_adapter, topN, exp_groups):
    basename = get_per_src_basename(per_adapter, topN, exp_groups=exp_groups)
    filename = os.path.join(exp_folder, basename)
    return read_df(exp_folder, filename)


def read_df(exp_folder, filename):
    df_path = os.path.join(exp_folder, filename)
    if not os.path.isfile(df_path):
        print(f"{df_path} not found, skipping")
        return None
    print(f"reading {df_path}")
    return pd.read_pickle(df_path)


def get_all_dfs(exp_folder):
    return read_df(exp_folder, ALL_DFS_FILENAME)


def get_processed_df(exp_folder):
    return read_df(exp_folder, PROCESSED_DF_FILENAME)


def tasks_match(e1, e2):
    return e1.split("_fl")[0] == e2.split("_fl")[0]


# combined across feature layers etc
def get_exp_groups_with_matching_tasks(exp_folder, exp_groups):
    num_exp_groups = len(exp_groups)
    combined_exp_groups = []
    for i in range(num_exp_groups):
        curr_exp_groups, curr_dfs = [], []
        e1 = exp_groups[i]
        if any(e1 in ceg for ceg in combined_exp_groups):
            continue
        df1 = get_processed_df(os.path.join(exp_folder, e1))

        for j in range(i + 1, num_exp_groups):
            e2 = exp_groups[j]
            if not tasks_match(e1, e2):
                continue
            df2 = get_processed_df(os.path.join(exp_folder, e2))
            if df1 is None or df2 is None:
                continue
            assert df1["task"].unique() == df2["task"].unique()
            curr_exp_groups.append(e2)

        if len(curr_exp_groups) > 0:
            combined_exp_groups.append((e1, *curr_exp_groups))

    return combined_exp_groups
