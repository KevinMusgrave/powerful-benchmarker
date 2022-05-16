import pandas as pd

from latex import utils as latex_utils
from latex.color_map_tags import absolute_value_greater_than, absolute_value_interval_fn
from latex.table_creator import table_creator
from validator_tests.utils.utils import validator_args_delimited


def get_preprocess_df(per_adapter):
    def fn(df):
        df = latex_utils.filter_validators(df)
        df["validator_args"] = df.apply(
            lambda x: validator_args_delimited(
                x["validator_args"], delimiter=" "
            ).replace("_", " "),
            axis=1,
        )
        if per_adapter:
            latex_utils.convert_adapter_column_names(df)
        return df

    return fn


def get_postprocess_df(per_adapter):
    def fn(df):
        df = pd.concat(df, axis=0).reset_index(drop=True)
        df = latex_utils.rename_validator_args(df)
        if per_adapter:
            df = df.groupby(["validator", "validator_args"]).mean()
        else:
            df = df.pivot(index=["validator", "validator_args"], columns="task")
            df = df.droplevel(0, axis=1)
            df = latex_utils.shortened_task_names(df)
        df = (df * 100).round(1)
        df.columns.names = (None,)
        df.index.names = (None, None)
        return df

    return fn


def correlation_src_threshold(args, threshold, per_adapter=False):
    per_adapter_str = "per_adapter_" if per_adapter else ""
    basename = f"correlation_{per_adapter_str}{threshold}_src_threshold"
    min_value_fn = lambda x: x.loc[("Accuracy", "Source Val")]
    max_value_fn = lambda _: 100
    operation_fn = absolute_value_greater_than
    interval_fn = absolute_value_interval_fn
    color_map_tag_kwargs = {
        "tag_prefix": f"{latex_utils.get_tag_prefix(basename)}",
        "min_value_fn": min_value_fn,
        "max_value_fn": max_value_fn,
        "operation_fn": operation_fn,
        "interval_fn": interval_fn,
        "num_steps": 11,
    }
    table_creator(
        args,
        basename,
        get_preprocess_df(per_adapter),
        get_postprocess_df(per_adapter),
        color_map_tag_kwargs,
        add_resizebox=True,
        clines="skip-last;data",
    )
