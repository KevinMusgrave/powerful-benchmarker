import pandas as pd

from latex import utils as latex_utils
from latex.table_creator import table_creator


def preprocess_df(df):
    latex_utils.convert_adapter_name(df)
    return df


def remove_accuracy_name_multiindex(df):
    accuracy_name = df.columns.levels[0]
    assert len(accuracy_name) == 2
    accuracy_name = [x for x in accuracy_name if not x.endswith("_std")]
    accuracy_name = accuracy_name[0]
    df = df[accuracy_name]
    return df, accuracy_name


def postprocess_df(df):
    df = pd.concat(df, axis=0)
    df = df.pivot(index="adapter", columns="task")
    df, accuracy_name = remove_accuracy_name_multiindex(df)
    df = latex_utils.add_source_only(df, accuracy_name)
    df = latex_utils.shortened_task_names(df)
    df = (df * 100).round(1)
    return df


def best_accuracy_per_adapter(args, topN):
    basename = f"best_accuracy_top{topN}_per_adapter"
    min_value_fn = lambda x: x.loc["Source only"]
    color_map_tag_kwargs = {
        "tag_prefix": latex_utils.get_tag_prefix(basename),
        "min_value_fn": min_value_fn,
    }
    caption = ("For each algorithm/task pair, we sorted model checkpoints by target accuracy, "
                f"and each cell in this table is the average accuracy of the top {topN} checkpoints. "
                "Green cells have an average accuracy greater than than the source-only model. "
                "A stronger green color indicates higher accuracy. The highest values per column are bolded.")
    table_creator(
        args,
        basename,
        preprocess_df,
        postprocess_df,
        color_map_tag_kwargs,
        add_resizebox=True,
        caption=caption,
    )
