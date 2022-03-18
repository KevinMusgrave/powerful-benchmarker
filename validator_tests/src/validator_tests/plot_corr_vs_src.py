import os

import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_adapt.utils import common_functions as c_f

from .plot_utils import plot_loop
from .utils import unify_validator_columns


def multiplot(
    plots_folder,
    df,
    x,
    y,
    filename,
    plot_fn,
    hue=None,
    rotation=0,
    order=None,
    xlim=None,
    ylim=None,
    other_kwargs=None,
    show_x_label=False,
):
    sns.set(font_scale=2, style="whitegrid", rc={"figure.figsize": (12.8, 9.6)})
    kwargs = {"data": df, "x": x, "y": y, "hue": hue}
    if order is not None:
        kwargs.update({"order": order})
    if other_kwargs:
        kwargs.update(other_kwargs)
    plot = plot_fn(**kwargs)
    plot.legend(loc="center left", bbox_to_anchor=(1, 0.5), title=hue)
    if not show_x_label:
        plot.set(xlabel="")
    plt.setp(plot.get_xticklabels(), rotation=rotation)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    fig = plot.get_figure()
    c_f.makedir_if_not_there(plots_folder)
    fig.savefig(
        os.path.join(plots_folder, f"{filename}.png"),
        bbox_inches="tight",
    )
    fig.clf()


def per_validator_args(threshold_type):
    def fn(curr_plots_folder, curr_df, filename):
        curr_df = unify_validator_columns(curr_df)
        multiplot(
            curr_plots_folder,
            curr_df,
            threshold_type,
            "correlation",
            filename,
            sns.lineplot,
            "validator",
        )

    return fn


def plot_corr_vs_X(threshold_type, per_adapter):
    def fn(df, plots_folder):
        plots_folder = os.path.join(plots_folder, f"corr_vs_{threshold_type}")

        filter_by = ["adapter"] if per_adapter else []

        plot_loop(
            df,
            plots_folder,
            per_validator_args(f"{threshold_type}_threshold"),
            filter_by=filter_by,
            sub_folder_components=[],
            filename_components=filter_by,
            filename=f"corr_vs_{threshold_type}",
            per_adapter=per_adapter,
        )

    return fn
