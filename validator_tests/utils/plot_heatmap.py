import os

import seaborn as sns
import tqdm
from pytorch_adapt.utils import common_functions as c_f

from .df_utils import unify_validator_columns


def plot_fn(df, plots_folder, filename, index, columns, values, **kwargs):
    sns.set(rc={"figure.figsize": (20, 15)})
    plots_folder = os.path.join(plots_folder, "heatmaps")
    c_f.makedir_if_not_there(plots_folder)

    df = df[[index, columns, values]]
    df = df.pivot(index, columns, values)

    plot = sns.heatmap(data=df, mask=df.isnull(), cmap="viridis", **kwargs)
    fig = plot.get_figure()
    fig.savefig(
        os.path.join(plots_folder, filename),
        bbox_inches="tight",
    )
    fig.clf()


def process_df(df):
    df = df[df["validator"] != "Accuracy"]
    return unify_validator_columns(df)


def plot_heatmap_per_adapter(df, plots_folder):
    df = process_df(df)
    print("plot_heatmap_per_adapter")
    for x in tqdm.tqdm([-0.01, 0.5, 0.9, 0.98, 0.99]):
        filename = f"heatmap_{x:.2f}.png"
        plot_fn(
            df[df["src_threshold"] == x],
            plots_folder,
            filename,
            "validator",
            "adapter",
            "correlation",
            vmin=-1,
            vmax=1,
        )

    df = (
        df.groupby(["validator", "adapter"])["correlation"]
        .mean()
        .reset_index(name="correlation")
    )
    plot_fn(
        df,
        plots_folder,
        "heatmap_avg_correlation.png",
        "validator",
        "adapter",
        "correlation",
        vmin=-1,
        vmax=1,
    )


def plot_heatmap(df, plots_folder):
    df = process_df(df)
    plot_fn(
        df,
        plots_folder,
        "heatmap_correlation_global.png",
        "validator",
        "src_threshold",
        "correlation",
        vmin=-1,
        vmax=1,
    )

    plot_fn(
        df,
        plots_folder,
        "heatmap_predicted_best_acc_global.png",
        "validator",
        "src_threshold",
        "predicted_best_acc",
    )
