import os

import seaborn as sns
import tqdm
from pytorch_adapt.utils import common_functions as c_f

from .df_utils import unify_validator_columns


def plot_fn(df, plots_folder, filename, index, columns, values, **kwargs):
    sns.set(rc={"figure.figsize": (40, 30)})
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


def plot_heatmap(df, plots_folder):
    print("plot_heatmap")
    df = process_df(df)
    plot_fn(
        df,
        plots_folder,
        "corr_global_heatmap.png",
        "validator",
        "src_threshold",
        "correlation",
        vmin=-1,
        vmax=1,
    )

    plot_fn(
        df,
        plots_folder,
        "predicted_acc_global_heatmap.png",
        "validator",
        "src_threshold",
        "predicted_best_acc",
        vmin=0,
        vmax=1,
    )


def plot_heatmap_per_adapter(df, plots_folder):
    df = process_df(df)
    print("plot_heatmap_per_adapter")
    for x in tqdm.tqdm([-0.01, 0.5, 0.9, 0.98, 0.99, 1]):
        curr_df = df[df["src_threshold"] == x]
        plot_fn(
            curr_df,
            plots_folder,
            f"corr_heatmap_{x:.2f}.png",
            "validator",
            "adapter",
            "correlation",
            vmin=-1,
            vmax=1,
        )
        plot_fn(
            curr_df,
            plots_folder,
            f"predicted_acc_heatmap_{x:.2f}.png",
            "validator",
            "adapter",
            "predicted_best_acc",
            vmin=0,
            vmax=1,
        )


def plot_heatmap_average_across_adapters(df, plots_folder):
    print("plot_heatmap_average_across_adapters")
    df = process_df(df)
    grouped = df.groupby(["validator", "src_threshold"])
    corr_df = grouped["correlation"].mean().reset_index(name="correlation")
    acc_df = grouped["predicted_best_acc"].mean().reset_index(name="predicted_best_acc")
    plot_fn(
        corr_df,
        plots_folder,
        "corr_avg_across_adapters_heatmap.png",
        "validator",
        "src_threshold",
        "correlation",
        vmin=-1,
        vmax=1,
    )

    plot_fn(
        acc_df,
        plots_folder,
        "predicted_acc_avg_across_adapters_heatmap.png",
        "validator",
        "src_threshold",
        "predicted_best_acc",
        vmin=0,
        vmax=1,
    )
