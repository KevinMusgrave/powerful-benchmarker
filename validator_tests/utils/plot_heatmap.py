import os

import seaborn as sns
import tqdm
from pytorch_adapt.utils import common_functions as c_f

from .df_utils import unify_validator_columns


def plot_at_one_threshold(df, plots_folder, threshold):
    sns.set(rc={"figure.figsize": (20, 15)})
    plots_folder = os.path.join(plots_folder, "heatmaps")
    c_f.makedir_if_not_there(plots_folder)

    df = df[df["src_threshold"] == threshold]
    df = df[["validator", "adapter", "correlation"]]
    df = df.pivot("validator", "adapter", "correlation")

    plot = sns.heatmap(data=df, mask=df.isnull(), cmap="viridis", vmin=-1, vmax=1)
    fig = plot.get_figure()
    fig.savefig(
        os.path.join(plots_folder, f"heatmap_{threshold:.2f}.png"),
        bbox_inches="tight",
    )
    fig.clf()


def plot_heatmap(df, plots_folder):
    df = df[df["validator"] != "Accuracy"]
    df = unify_validator_columns(df)
    print("plot_heatmap")
    for x in tqdm.tqdm([-0.01, 0.5, 0.9, 0.98, 0.99]):
        plot_at_one_threshold(df, plots_folder, x)
