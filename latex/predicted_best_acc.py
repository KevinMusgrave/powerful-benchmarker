from latex import utils as latex_utils
from latex.color_map_tags import default_interval_fn, reverse_interval_fn
from latex.correlation_src_threshold import get_postprocess_df, get_preprocess_df
from latex.table_creator import table_creator


def std_condition(std, c):
    endwith_std = c.endswith("_std")
    return (std and endwith_std) or (not std and not endwith_std)


def get_preprocess_df_wrapper(per_adapter, std=False):
    fn2 = get_preprocess_df(per_adapter)

    def fn(df):

        if per_adapter:
            for c in df.columns.levels[0]:
                if std_condition(std, c):
                    df = df[c].reset_index()
                    break
        else:
            columns = []
            for c in df.columns:
                if (not c.startswith("predicted_best_acc")) or std_condition(std, c):
                    columns.append(c)
            df = df[columns]
        return fn2(df)

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


def predicted_best_acc(args, topN, threshold, per_adapter=False):
    per_adapter_str = "per_adapter_" if per_adapter else ""
    basename = (
        f"predicted_best_acc_top{topN}_{per_adapter_str}{threshold}_src_threshold"
    )
    color_map_tag_kwargs = {
        "tag_prefix": latex_utils.get_tag_prefix(basename),
        "min_value_fn": min_value_fn,
        "max_value_fn": max_value_fn,
        "num_steps": 11,
        "interval_fn": interval_fn,
        "operation_fn": operation_fn,
    }

    threshold_str = int(threshold * 100)
    caption = (
        f"The Top {topN} RTA of each validator/task pair, after removing checkpoints with < {threshold_str}\% RSVA. "
        "Green cells have better performance than the Source Val Accuracy validator."
    )

    if per_adapter:
        highlight_max_subset = latex_utils.adapter_names()
    else:
        highlight_max_subset = list(latex_utils.shortened_task_name_dict().values())
    highlight_max_subset += ["Mean"]

    final_str_hook = (
        latex_utils.validator_per_adapter_final_str_hook
        if per_adapter
        else latex_utils.validator_final_str_hook
    )

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
