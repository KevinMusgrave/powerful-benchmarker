import pandas as pd

from latex import utils as latex_utils
from latex.correlation import (
    base_filename,
    get_add_resizebox,
    get_caption,
    get_highlight_max_subset,
    get_highlight_min,
    get_highlight_min_subset,
    get_label_prefix,
    get_preprocess_df,
    interval_fn,
    max_value_fn,
    min_value_fn,
    operation_fn,
)
from latex.table_creator import table_creator


def get_postprocess_df(remove_index_names=True):
    def fn(df):
        df = pd.concat(df, axis=0).reset_index(drop=True)
        df = latex_utils.rename_validator_args(df)
        df = df.pivot(index=["validator", "validator_args"], columns="task")
        df_dict = {}
        for i in df.columns.levels[0]:
            df_dict[i] = df[i]
        for k, df in df_dict.items():
            df = latex_utils.shortened_task_names(df)
            df = latex_utils.add_mean_std_column(df)
            df = (df * 100).round(1)
            df.columns.names = (None,)
            if remove_index_names:
                df.index.names = (None, None)
            df_dict[k] = df
        return df_dict

    return fn


def caption_hook(caption, k):
    return caption.replace("pair", f"pair for \\textbf{{{k}}}")


def correlation_single_adapter(args, name, src_threshold):
    basename = base_filename(name, True, src_threshold)
    color_map_tag_kwargs = {
        "tag_prefix": latex_utils.get_tag_prefix(basename),
        "min_value_fn": min_value_fn,
        "max_value_fn": max_value_fn,
        "num_steps": 11,
        "interval_fn": interval_fn,
        "operation_fn": operation_fn,
    }

    caption = get_caption(per_adapter=False, short_caption=True)

    highlight_max_subset = get_highlight_max_subset(per_adapter=False)
    highlight_min_subset = get_highlight_min_subset()

    table_creator(
        args,
        args.input_folder,
        args.output_folder,
        basename,
        get_preprocess_df(per_adapter=True),
        get_postprocess_df(),
        color_map_tag_kwargs,
        add_resizebox=get_add_resizebox(args),
        clines="skip-last;data",
        caption=caption,
        highlight_min=get_highlight_min(args),
        highlight_max_subset=highlight_max_subset,
        highlight_min_subset=highlight_min_subset,
        final_str_hook=latex_utils.validator_final_str_hook,
        caption_hook=caption_hook,
        position="H",
        label_prefix=get_label_prefix(args),
    )
