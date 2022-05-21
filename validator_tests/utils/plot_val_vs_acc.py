import os

import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_adapt.utils import common_functions as c_f

from powerful_benchmarker.utils.score_utils import pretrained_src_accuracy

from . import df_utils, threshold_utils
from .constants import TARGET_ACCURACY
from .plot_utils import plot_loop


def scatter_plot(
    plots_folder,
    df,
    x,
    y,
    filename,
    c=None,
    colobar_label=None,
    log_x=False,
    s=0.1,
    font_scale=0.8,
    figsize=(4.8, 4.8),
    show_x_label=True,
    show_y_label=True,
    colorbar=True,
    x_label=None,
    y_label=None,
):
    sns.set(font_scale=font_scale, style="whitegrid", rc={"figure.figsize": figsize})
    if colorbar:
        points = plt.scatter(
            df[x],
            df[y],
            c=df[c] if c is not None else None,
            s=s,
            cmap="rainbow",
        )
        if c:
            cbar = plt.colorbar(points)
        if colobar_label:
            cbar.set_label(colobar_label)
        if show_x_label:
            plt.xlabel(x if x_label is None else x_label)
        if show_y_label:
            plt.ylabel(y if y_label is None else y_label)
        if log_x:
            plt.xscale("symlog")
        fig = plt
    else:
        plot = sns.scatterplot(data=df, x=x, y=y, hue=c, s=s)
        fig = plot.get_figure()
    c_f.makedir_if_not_there(plots_folder)
    fig.savefig(
        os.path.join(plots_folder, f"{filename}.png"),
        bbox_inches="tight",
    )
    fig.clf()


def get_score_vs_target_accuracy_fn(**kwargs):
    def fn(curr_plots_folder, curr_df, filename):
        input_kwargs = {
            "plots_folder": curr_plots_folder,
            "df": curr_df,
            "x": "score",
            "y": TARGET_ACCURACY,
            "filename": filename,
            "c": "src_val_micro",
            "x_label": "Validation Score",
            "y_label": "Target Accuracy",
        }
        input_kwargs.update(kwargs)
        scatter_plot(**input_kwargs)

    return fn


def plot_val_vs_acc(
    df,
    plots_folder,
    per_adapter,
    per_feature_layer,
    validator_set=None,
    src_threshold=None,
    **kwargs,
):
    plots_folder = os.path.join(plots_folder, "val_vs_acc")

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
        dataset = df_utils.get_sorted_unique(df, "dataset", assert_one=True)[0]
        src_domains = df_utils.get_sorted_unique(df, "src_domains", assert_one=True)[0]
        min_acc = pretrained_src_accuracy(dataset, src_domains, "val", "micro")
        df = threshold_utils.filter_by_acc(df, min_acc * src_threshold, "src")
        filename_suffix = f"{src_threshold}_src_threshold"

    plot_loop(
        df,
        plots_folder,
        get_score_vs_target_accuracy_fn(**kwargs),
        filter_by=filter_by,
        sub_folder_components=sub_folder_components,
        filename_components=["validator", "validator_args"],
        filename_suffix=filename_suffix,
        per_adapter=per_adapter,
        validator_set=validator_set,
    )
