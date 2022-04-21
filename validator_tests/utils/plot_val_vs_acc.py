import os

import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_adapt.utils import common_functions as c_f

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
    font_scale=2,
    figsize=(12.8, 9.6),
    show_x_label=True,
    show_y_label=True,
    colorbar=True,
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
            plt.xlabel(x)
        if show_y_label:
            plt.ylabel(y)
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


def score_vs_target_accuracy(curr_plots_folder, curr_df, filename):
    scatter_plot(
        curr_plots_folder,
        curr_df,
        "score",
        TARGET_ACCURACY,
        filename,
        "src_val_macro",
    )


def plot_val_vs_acc(
    df, plots_folder, per_adapter, per_feature_layer, validator_set=None
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

    plot_loop(
        df,
        plots_folder,
        score_vs_target_accuracy,
        filter_by=filter_by,
        sub_folder_components=sub_folder_components,
        filename_components=["validator", "validator_args"],
        per_adapter=per_adapter,
        validator_set=validator_set,
    )
