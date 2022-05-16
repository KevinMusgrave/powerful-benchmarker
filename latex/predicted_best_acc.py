from latex import utils as latex_utils
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


def predicted_best_acc(args, topN, threshold, per_adapter=False):
    per_adapter_str = "per_adapter_" if per_adapter else ""
    basename = (
        f"predicted_best_acc_top{topN}_{per_adapter_str}{threshold}_src_threshold"
    )
    min_value_fn = lambda x: x.loc[("Accuracy", "Source Val")]
    # max_value_fn = lambda _: 100
    color_map_tag_kwargs = {
        "tag_prefix": latex_utils.get_tag_prefix(basename),
        "min_value_fn": min_value_fn,
        # "max_value_fn": max_value_fn,
        "num_steps": 11,
    }

    caption = (
        f"The average relative accuracy of the top {topN} checkpoints selected by each validator, "
        "after removing all checkpoints that have less than 87\\% relative source validation accuracy."
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
    )
