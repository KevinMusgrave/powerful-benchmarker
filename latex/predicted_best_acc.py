from latex import utils as latex_utils
from latex.correlation_src_threshold import get_postprocess_df, get_preprocess_df
from latex.table_creator import table_creator


def predicted_best_acc(args, topN, threshold, per_adapter=False):
    per_adapter_str = "per_adapter_" if per_adapter else ""
    basename = (
        f"predicted_best_acc_top{topN}_{per_adapter_str}{threshold}_src_threshold"
    )
    min_value_fn = lambda _: 80
    max_value_fn = lambda _: 100
    color_map_tag_kwargs = {
        "tag_prefix": latex_utils.get_tag_prefix(basename),
        "min_value_fn": min_value_fn,
        "max_value_fn": max_value_fn,
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
