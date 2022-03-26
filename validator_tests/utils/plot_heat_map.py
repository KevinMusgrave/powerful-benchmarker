import os

import seaborn as sns
from pytorch_adapt.utils import common_functions as c_f

from .df_utils import unify_validator_columns


def plot_heat_map(df, plots_folder):
    plots_folder = os.path.join(plots_folder, "heatmaps")
    c_f.makedir_if_not_there(plots_folder)

    df = df[df["validator"] != "Accuracy"]
    df = df[df["src_threshold"] == -0.01]
    df = unify_validator_columns(df)
    df = df[["validator", "adapter", "correlation"]]
    df = df.pivot("validator", "adapter", "correlation")

    plot = sns.heatmap(data=df)
    fig = plot.get_figure()
    fig.savefig(
        os.path.join(plots_folder, f"heatmap.png"),
        bbox_inches="tight",
    )
    fig.clf()
