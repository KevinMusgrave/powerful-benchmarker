import pandas as pd

from latex import utils as latex_utils
from latex.table_creator import save_to_latex, table_creator


def get_preprocess_df(per_adapter):
    def fn(df):
        if per_adapter:
            latex_utils.convert_adapter_name(df)
        return df

    return fn


def get_postprocess_df(per_adapter):
    def fn(df):
        df = pd.concat(df, axis=0).reset_index(drop=True)
        if not per_adapter:
            df["adapter"] = "All"
        df = df[["adapter", "task", "src_threshold"]]
        df = df.pivot(index="adapter", columns="task")
        df.columns.names = (None, None)
        df = df["src_threshold"]
        df = latex_utils.shortened_task_names(df)
        return df

    return fn


def highest_src_threshold_possible(args, topN, topN_per_adapter):
    basename = f"highest_src_threshold_possible_top{topN}"
    df1, _ = table_creator(
        args,
        basename,
        get_preprocess_df(False),
        get_postprocess_df(False),
        do_save_to_latex=False,
    )

    basename = f"highest_src_threshold_possible_top{topN_per_adapter}_per_adapter"
    df2, output_folder = table_creator(
        args,
        basename,
        get_preprocess_df(True),
        get_postprocess_df(True),
        do_save_to_latex=False,
    )

    df = pd.concat([df1, df2], axis=0)
    df = (df * 100).astype(int)

    basename = (
        f"highest_src_threshold_possible_top{topN}_top{topN_per_adapter}_per_adapter"
    )

    caption = (
        "The highest source validation accuracy that can be used as a filter, "
        "without removing the top target train set accuracy. "
        "Smallest values per task are bolded."
    )

    save_to_latex(
        df,
        output_folder,
        basename,
        color_map_tag_kwargs=None,
        add_resizebox=True,
        clines=None,
        highlight_max=False,
        caption=caption,
        highlight_min=True,
        label=basename,
    )
