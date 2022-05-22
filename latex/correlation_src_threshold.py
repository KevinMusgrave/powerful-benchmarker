from latex import utils as latex_utils
from latex.color_map_tags import absolute_value_greater_than, absolute_value_interval_fn
from latex.predicted_best_acc import (
    get_final_str_hook,
    get_highlight_max_subset,
    get_postprocess_df,
    get_preprocess_df,
    interval_fn,
    max_value_fn,
    min_value_fn,
    operation_fn,
)
from latex.table_creator import table_creator


def operation_fn_wrapper(lower_bound, column_name):
    if column_name != "Std":
        return absolute_value_greater_than(lower_bound, column_name)
    return operation_fn(lower_bound, column_name)


def interval_fn_wrapper(min_value, max_value, num_steps, column_name):
    if column_name != "Std":
        return absolute_value_interval_fn(min_value, max_value, num_steps, column_name)
    return interval_fn(min_value, max_value, num_steps, column_name)


def correlation_src_threshold(args, threshold, per_adapter=False):
    per_adapter_str = "per_adapter_" if per_adapter else ""
    basename = f"correlation_{per_adapter_str}{threshold}_src_threshold"
    color_map_tag_kwargs = {
        "tag_prefix": f"{latex_utils.get_tag_prefix(basename)}",
        "min_value_fn": min_value_fn,
        "max_value_fn": max_value_fn,
        "operation_fn": operation_fn_wrapper,
        "interval_fn": interval_fn_wrapper,
        "num_steps": 11,
    }

    highlight_max_subset = get_highlight_max_subset(per_adapter)
    final_str_hook = get_final_str_hook(per_adapter)

    table_creator(
        args,
        basename,
        get_preprocess_df(per_adapter),
        get_postprocess_df(per_adapter),
        color_map_tag_kwargs,
        add_resizebox=True,
        clines="skip-last;data",
        highlight_min=True,
        highlight_max_subset=highlight_max_subset,
        highlight_min_subset=["Std"],
        final_str_hook=final_str_hook,
    )
