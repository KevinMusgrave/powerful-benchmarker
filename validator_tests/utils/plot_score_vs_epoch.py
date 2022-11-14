import os

import seaborn as sns
from pytorch_adapt.utils import common_functions as c_f

from .plot_utils import filter_and_plot


def line_plot(
    plots_folder,
    df,
    x,
    y,
    filename,
    font_scale=0.8,
    figsize=(4.8, 4.8),
):
    sns.set(font_scale=font_scale, style="whitegrid", rc={"figure.figsize": figsize})
    plot = sns.lineplot(data=df, x=x, y=y)
    fig = plot.get_figure()
    c_f.makedir_if_not_there(plots_folder)
    fig.savefig(
        os.path.join(plots_folder, f"{filename}.png"),
        bbox_inches="tight",
    )
    fig.clf()


def get_score_vs_epoch_fn(**kwargs):
    def fn(curr_plots_folder, curr_df, filename):
        curr_df = curr_df.astype({"epoch": int})
        input_kwargs = {
            "plots_folder": curr_plots_folder,
            "df": curr_df,
            "x": "epoch",
            "y": "score",
            "filename": filename,
        }
        input_kwargs.update(kwargs)
        line_plot(**input_kwargs)

    return fn


def plot_score_vs_epoch(
    df,
    plots_folder,
    per_adapter,
    per_feature_layer,
    validator_set=None,
    src_threshold=None,
    adapter=None,
    **kwargs,
):
    plots_folder = os.path.join(plots_folder, "score_vs_epoch")

    filter_and_plot(
        df,
        get_score_vs_epoch_fn(**kwargs),
        plots_folder,
        per_adapter,
        per_feature_layer,
        validator_set,
        src_threshold,
        adapter,
    )
