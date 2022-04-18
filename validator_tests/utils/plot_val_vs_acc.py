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


def per_validator_args_per_task(curr_plots_folder, curr_df, filename):
    scatter_plot(
        curr_plots_folder,
        curr_df,
        "score",
        TARGET_ACCURACY,
        filename,
        "src_val_macro",
    )


def per_adapter_per_validator_args_per_task(curr_plots_folder, curr_df, filename):
    scatter_plot(
        curr_plots_folder,
        curr_df,
        "score",
        TARGET_ACCURACY,
        filename,
        "src_val_macro",
    )


def plot_val_vs_acc(df, plots_folder, per_adapter, validator_set=None):
    plots_folder = os.path.join(plots_folder, "val_vs_acc")

    if not per_adapter:
        plot_loop(
            df,
            plots_folder,
            per_validator_args_per_task,
            filter_by=[
                "dataset",
                "src_domains",
                "target_domains",
                "validator",
                "validator_args",
            ],
            sub_folder_components=["dataset", "src_domains", "target_domains"],
            filename_components=["validator", "validator_args"],
            per_adapter=per_adapter,
            validator_set=validator_set,
        )

    else:
        plot_loop(
            df,
            plots_folder,
            per_adapter_per_validator_args_per_task,
            filter_by=[
                "adapter",
                "dataset",
                "src_domains",
                "target_domains",
                "validator",
                "validator_args",
            ],
            sub_folder_components=[
                "dataset",
                "src_domains",
                "target_domains",
                "validator",
                "adapter",
            ],
            filename_components=["validator_args"],
            per_adapter=per_adapter,
            validator_set=validator_set,
        )
