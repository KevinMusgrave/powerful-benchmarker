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
    # basename = f"highest_src_threshold_possible_top{topN}"
    # df1, _ = table_creator(
    #     args,
    #     basename,
    #     get_preprocess_df(False),
    #     get_postprocess_df(False),
    #     do_save_to_latex=False,
    # )

    basename = f"highest_src_threshold_possible_top{topN_per_adapter}_per_adapter"
    df, output_folder = table_creator(
        args,
        basename,
        get_preprocess_df(True),
        get_postprocess_df(True),
        do_save_to_latex=False,
    )

    # df = pd.concat([df1, df2], axis=0)
    df = (df * 100).astype(int)
    df.index.name = None

    basename = f"highest_src_threshold_possible_top{topN_per_adapter}_per_adapter"

    caption = (
        "For each algorithm/task pair, we raised the relative source validation accuracy threshold as high "
        f"as possible, without removing any of the top {topN_per_adapter} checkpoints (sorted by target accuracy). "
        "For example, for the GVB/WD pair, all checkpoints with relative accuracy less than 73\% can be discarded "
        f"without losing any of the top {topN_per_adapter} target accuracies. "
        f'The ``All" row uses the top {topN} checkpoints from all algorithms grouped together.'
    )

    save_to_latex(
        df,
        output_folder,
        basename,
        color_map_tag_kwargs=None,
        add_resizebox=True,
        clines=None,
        highlight_max=False,
        highlight_min=True,
        caption=caption,
        label=basename,
        final_str_hook=latex_utils.adapter_final_str_hook,
    )
