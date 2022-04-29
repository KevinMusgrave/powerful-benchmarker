import pandas as pd

from latex import utils as latex_utils
from latex.table_creator import table_creator


def preprocess_df(df):
    return df


def remove_accuracy_name_multiindex(df):
    accuracy_name = df.columns.levels[0]
    assert len(accuracy_name) == 1
    accuracy_name = accuracy_name[0]
    df = df.droplevel(0, axis=1)
    return df, accuracy_name


def postprocess_df(df):
    df = pd.concat(df, axis=0)
    print(df)
    df = df.pivot(index=["validator", "validator_args"], columns="task")
    df = latex_utils.shortened_task_names(df)
    df = (df * 100).round(1)
    return df


def correlation_src_threshold(args, threshold):
    basename = f"correlation_{threshold}_src_threshold"
    table_creator(args, basename, preprocess_df, postprocess_df)
