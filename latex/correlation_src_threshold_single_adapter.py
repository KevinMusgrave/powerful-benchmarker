from latex import utils as latex_utils
from latex.correlation_src_threshold import (
    get_caption,
    interval_fn_wrapper,
    operation_fn_wrapper,
)
from latex.predicted_best_acc import get_preprocess_df, max_value_fn, min_value_fn
from latex.predicted_best_acc_single_adapter import caption_hook, postprocess_df
from latex.table_creator import table_creator


def correlation_src_threshold_single_adapter(args, threshold):
    basename = f"correlation_per_adapter_{threshold}_src_threshold"
    color_map_tag_kwargs = {
        "tag_prefix": latex_utils.get_tag_prefix(basename),
        "min_value_fn": min_value_fn,
        "max_value_fn": max_value_fn,
        "num_steps": 11,
        "interval_fn": interval_fn_wrapper,
        "operation_fn": operation_fn_wrapper,
    }

    caption = get_caption(threshold, per_adapter=False)

    highlight_max_subset = list(latex_utils.shortened_task_name_dict().values()) + [
        "Mean"
    ]

    table_creator(
        args,
        basename,
        get_preprocess_df(per_adapter=True),
        postprocess_df,
        color_map_tag_kwargs,
        add_resizebox=True,
        clines="skip-last;data",
        caption=caption,
        highlight_min=True,
        highlight_max_subset=highlight_max_subset,
        highlight_min_subset=["Std"],
        final_str_hook=latex_utils.validator_final_str_hook,
        caption_hook=caption_hook,
        position="H",
    )
