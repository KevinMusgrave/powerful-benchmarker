import pandas as pd

from latex import utils as latex_utils
from latex.color_map_tags import absolute_value_greater_than, absolute_value_interval_fn
from latex.table_creator import table_creator
from validator_tests.utils.utils import validator_args_delimited


def preprocess_df(df):
    df["validator_args"] = df.apply(
        lambda x: validator_args_delimited(x["validator_args"], delimiter=" ").replace(
            "_", " "
        ),
        axis=1,
    )
    return df


def remove_correlation_multiindex(df):
    df = df.droplevel(0, axis=1)
    return df


def postprocess_df(df):
    df = pd.concat(df, axis=0)
    print(df)
    df = df.pivot(index=["validator", "validator_args"], columns="task")
    df = remove_correlation_multiindex(df)
    df = latex_utils.shortened_task_names(df)
    df = (df * 100).round(1)
    return df


def get_tag_prefix(basename):
    num_to_word = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]
    basename = basename.replace("_", "").replace(".", "")
    for i in range(10):
        basename = basename.replace(str(i), num_to_word[i])
    return basename


def correlation_src_threshold(args, threshold):
    basename = f"correlation_{threshold}_src_threshold"
    min_value_fn = lambda _: 10
    max_value_fn = lambda _: 100
    operation_fn = absolute_value_greater_than
    interval_fn = absolute_value_interval_fn
    color_map_tag_kwargs = {
        "tag_prefix": f"{get_tag_prefix(basename)}",
        "min_value_fn": min_value_fn,
        "max_value_fn": max_value_fn,
        "operation_fn": operation_fn,
        "interval_fn": interval_fn,
        "num_steps": 10,
    }
    table_creator(args, basename, preprocess_df, postprocess_df, color_map_tag_kwargs)
