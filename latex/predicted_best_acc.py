from latex import utils as latex_utils
from latex.correlation_src_threshold import postprocess_df, preprocess_df
from latex.table_creator import table_creator


def predicted_best_acc(args, topN, threshold):
    basename = f"predicted_best_acc_top{topN}_{threshold}_src_threshold"
    min_value_fn = lambda _: 50
    max_value_fn = lambda _: 100
    color_map_tag_kwargs = {
        "tag_prefix": f"{latex_utils.get_tag_prefix(basename)}",
        "min_value_fn": min_value_fn,
        "max_value_fn": max_value_fn,
        "num_steps": 11,
    }
    table_creator(
        args,
        basename,
        preprocess_df,
        postprocess_df,
        color_map_tag_kwargs,
        add_resizebox=True,
    )
