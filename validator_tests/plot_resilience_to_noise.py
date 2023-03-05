import glob
import os

import pandas as pd
import seaborn as sns


def save_plot(output_folder, df):
    df = pd.melt(
        df,
        id_vars=["Noise Standard Deviation"],
        value_vars=[
            "Weighted Spearman Correlation",
            "Top 1 Accuracy",
            "Top 5 Accuracy",
        ],
        var_name="Metric",
        value_name="Correlation with original data",
    )

    plot = sns.lineplot(
        data=df,
        x="Noise Standard Deviation",
        y="Correlation with original data",
        hue="Metric",
    )
    fig = plot.get_figure()
    fig.savefig(
        os.path.join(output_folder, "resilience.png"),
        bbox_inches="tight",
    )
    fig.clf()


def main():
    folder = "plots/resilience_to_noise"
    folders = glob.glob(os.path.join(folder, "*"))

    for f in folders:
        df = pd.read_pickle(os.path.join(f, "df.pkl"))
        save_plot(f, df)


main()
