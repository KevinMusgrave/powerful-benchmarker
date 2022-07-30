from latex import utils as latex_utils
from latex.predicted_best_acc import (
    get_final_str_hook,
    get_highlight_max_subset,
    get_postprocess_df,
    get_preprocess_df_wrapper,
    interval_fn,
    max_value_fn,
    min_value_fn,
    operation_fn,
)
from latex.table_creator import table_creator


def averaged_predicted_best_acc(args, topN_list, threshold, per_adapter=False):
    per_adapter_str = "per_adapter_" if per_adapter else ""
    topN_list = [str(x) for x in topN_list]
    basename = f"averaged_predicted_best_acc_top{'_'.join(topN_list)}_{per_adapter_str}{threshold}_src_threshold"
    color_map_tag_kwargs = {
        "tag_prefix": latex_utils.get_tag_prefix(basename),
        "min_value_fn": min_value_fn,
        "max_value_fn": max_value_fn,
        "num_steps": 11,
        "interval_fn": interval_fn,
        "operation_fn": operation_fn,
    }

    caption = ""
    highlight_max_subset = get_highlight_max_subset(per_adapter)
    final_str_hook = get_final_str_hook(per_adapter)

    table_creator(
        args,
        basename,
        get_preprocess_df_wrapper(per_adapter),
        get_postprocess_df(per_adapter),
        color_map_tag_kwargs,
        add_resizebox=True,
        clines="skip-last;data",
        caption=caption,
        highlight_min=True,
        highlight_max_subset=highlight_max_subset,
        highlight_min_subset=["Std"],
        final_str_hook=final_str_hook,
    )
