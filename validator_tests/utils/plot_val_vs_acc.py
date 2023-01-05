import os

import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_adapt.utils import common_functions as c_f

from .constants import TARGET_ACCURACY
from .plot_utils import filter_and_plot


def _scatter_plot(
    df,
    x,
    y,
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
    alpha=None,
    cmap="rainbow",
    invert_cmap_axis=False,
):
    sns.set(font_scale=font_scale, style="whitegrid", rc={"figure.figsize": figsize})
    if colorbar:
        points = plt.scatter(
            df[x],
            df[y],
            c=df[c] if c is not None else None,
            s=s,
            cmap=cmap,
            alpha=alpha,
        )
        if c:
            cbar = plt.colorbar(points)
            if invert_cmap_axis:
                cbar.ax.invert_yaxis()
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
        plot = sns.scatterplot(data=df, x=x, y=y, hue=c, s=s, alpha=alpha)
        fig = plot.get_figure()
    return fig


def scatter_plot(plots_folder, df, x, y, filename, **kwargs):
    fig = _scatter_plot(df, x, y, **kwargs)
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
    adapter=None,
    **kwargs,
):
    plots_folder = os.path.join(plots_folder, "val_vs_acc")

    filter_and_plot(
        df,
        get_score_vs_target_accuracy_fn(**kwargs),
        plots_folder,
        per_adapter,
        per_feature_layer,
        validator_set,
        src_threshold,
        adapter,
    )
