import re

import numpy as np
import pandas as pd

from latex import utils as latex_utils
from latex.color_map_tags import default_interval_fn, reverse_interval_fn
from latex.table_creator import table_creator
from validator_tests.utils.utils import validator_args_delimited


def filter_and_process_validator_args(df):
    df = latex_utils.filter_validators(df)
    df["validator_args"] = df.apply(
        lambda x: validator_args_delimited(x["validator_args"], delimiter=" ").replace(
            "_", " "
        ),
        axis=1,
    )
    return df


def get_preprocess_df(per_adapter):
    def fn(df):
        df = filter_and_process_validator_args(df)
        if per_adapter:
            latex_utils.convert_adapter_column_names(df)
        return df

    return fn


def get_postprocess_df(per_adapter, remove_index_names=True):
    def fn(df):
        df = pd.concat(df, axis=0).reset_index(drop=True)
        df = latex_utils.rename_validator_args(df)
        if per_adapter:
            df = df.groupby(["validator", "validator_args"], dropna=False).agg(np.mean)
        else:
            df = df.pivot(index=["validator", "validator_args"], columns="task")
            df = df.droplevel(0, axis=1)
            df = latex_utils.shortened_task_names(df)
        df = latex_utils.add_mean_std_column(df)
        df = (df * 100).round(1)
        df.columns.names = (None,)
        if remove_index_names:
            df.index.names = (None, None)
        return df

    return fn


def interval_fn(min_value, max_value, num_steps, column_name):
    if column_name == "Std":
        return reverse_interval_fn(min_value, max_value, num_steps, column_name)
    return default_interval_fn(min_value, max_value, num_steps, column_name)


def operation_fn(lower_bound, column_name):
    if column_name == "Std":
        return "<"
    return ">"


def min_value_fn(curr_df, column_name):
    if column_name == "Std":
        return curr_df.min()
    return curr_df.loc[("Accuracy", "Source Val")]


def max_value_fn(curr_df, column_name):
    if column_name == "Std":
        return curr_df.loc[("Accuracy", "Source Val")]
    return curr_df.max()


def get_highlight_max_subset(per_adapter):
    def fn(df):
        if per_adapter:
            highlight_max_subset = latex_utils.adapter_names()
        else:
            highlight_max_subset = list(
                set(latex_utils.shortened_task_name_dict().values()).intersection(
                    set(df.columns.values)
                )
            )
        if "Mean" in df.columns:
            highlight_max_subset += ["Mean"]
        return highlight_max_subset

    return fn


def get_highlight_min_subset():
    def fn(df):
        if "Std" in df.columns:
            return ["Std"]
        return None

    return fn


def get_final_str_hook(per_adapter):
    return (
        latex_utils.validator_per_adapter_final_str_hook
        if per_adapter
        else latex_utils.validator_final_str_hook
    )


def remove_whitespace_before_punctuation(x):
    return re.sub(r'\s+([?.!",](?:\s|$))', r"\1", x)


def get_caption(per_adapter, short_caption=False):
    pair_str = "algorithm" if per_adapter else "task"
    caption = f"The weighted Spearman correlation of each validator/{pair_str} pair."
    if not short_caption:
        mean_std_str = "algorithm" if per_adapter else "task"
        caption += (
            " Green cells have better performance than the Source Val Accuracy validator. "
            f"The best value per column is bolded. The Mean and Std columns are the mean and standard deviation of all {mean_std_str} columns."
        )

    # https://stackoverflow.com/a/18878970
    return remove_whitespace_before_punctuation(caption)


def base_filename(name, per_adapter, src_threshold):
    per_adapter_str = "_per_adapter" if per_adapter else ""
    return f"{name}_{src_threshold}_src_threshold{per_adapter_str}"


def get_add_resizebox(args):
    return args.exp_group_prefix != "mnist"


def get_highlight_min(args):
    return args.exp_group_prefix != "mnist"


def get_label_prefix(args):
    if args.exp_group_prefix:
        return f"{args.exp_group_prefix}_"
    return ""


def correlation(args, per_adapter, name, src_threshold):
    basename = base_filename(name, per_adapter, src_threshold)
    color_map_tag_kwargs = {
        "tag_prefix": latex_utils.get_tag_prefix(basename),
        "min_value_fn": min_value_fn,
        "max_value_fn": max_value_fn,
        "num_steps": 11,
        "interval_fn": interval_fn,
        "operation_fn": operation_fn,
    }

    caption = get_caption(per_adapter)
    highlight_max_subset = get_highlight_max_subset(per_adapter)
    highlight_min_subset = get_highlight_min_subset()
    final_str_hook = get_final_str_hook(per_adapter)

    table_creator(
        args,
        args.input_folder,
        args.output_folder,
        basename,
        get_preprocess_df(per_adapter),
        get_postprocess_df(per_adapter),
        color_map_tag_kwargs,
        add_resizebox=get_add_resizebox(args),
        clines="skip-last;data",
        caption=caption,
        highlight_min=get_highlight_min(args),
        highlight_max_subset=highlight_max_subset,
        highlight_min_subset=highlight_min_subset,
        final_str_hook=final_str_hook,
        position="H",
        label_prefix=get_label_prefix(args),
    )
