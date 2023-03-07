import pandas as pd

from latex import utils as latex_utils
from latex.table_creator import table_creator
from validator_tests.utils.constants import TARGET_ACCURACY


def preprocess_df(df):
    latex_utils.convert_adapter_name(df)
    return df


def reshape_into_best_accuracy_table(df):
    df = df.pivot(index="adapter", columns="task").droplevel(level=0, axis=1)
    df = latex_utils.add_source_only(df, TARGET_ACCURACY)
    df = latex_utils.shortened_task_names(df)
    df = (df * 100).round(1)
    return df


def postprocess_df(df):
    df = pd.concat(df, axis=0)
    df = df.drop(columns=[f"{TARGET_ACCURACY}_std"])
    return reshape_into_best_accuracy_table(df)


def min_value_fn(x, *_):
    return x.loc["Source only"]


def best_accuracy_per_adapter(args):
    nlargest = args.nlargest
    basename = f"best_accuracy_per_adapter_{nlargest}"
    color_map_tag_kwargs = {
        "tag_prefix": latex_utils.get_tag_prefix(basename),
        "min_value_fn": min_value_fn,
    }
    caption = (
        f"The average of the top {nlargest} target domain accuracies per adapter/task pair. "
        "Green cells have an average accuracy greater than than the source-only model. "
        "A stronger green color indicates higher accuracy. The highest value per column is bolded."
    )
    return table_creator(
        args,
        args.input_folder,
        args.output_folder,
        basename,
        preprocess_df,
        postprocess_df,
        color_map_tag_kwargs,
        add_resizebox=True,
        caption=caption,
        final_str_hook=latex_utils.adapter_final_str_hook,
    )
