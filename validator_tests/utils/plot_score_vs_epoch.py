import os

import seaborn as sns
from pytorch_adapt.utils import common_functions as c_f

from . import threshold_utils
from .plot_utils import plot_loop


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

    filter_by = [
        "dataset",
        "src_domains",
        "target_domains",
        "validator",
        "validator_args",
    ]

    sub_folder_components = ["dataset", "src_domains", "target_domains"]

    if per_adapter:
        filter_by.append("adapter")
        sub_folder_components.append("adapter")
    if per_feature_layer:
        filter_by.append("feature_layer")
        sub_folder_components.append("feature_layer")

    filename_suffix = ""
    if src_threshold is not None:
        df = threshold_utils.filter_by_src_threshold(
            df, src_threshold, filter_action="remove"
        )
        filename_suffix = f"{src_threshold}_src_threshold"

    plot_loop(
        df,
        plots_folder,
        get_score_vs_epoch_fn(**kwargs),
        filter_by=filter_by,
        sub_folder_components=sub_folder_components,
        filename_components=["validator", "validator_args"],
        filename_suffix=filename_suffix,
        per_adapter=per_adapter,
        validator_set=validator_set,
        adapter=adapter,
    )
