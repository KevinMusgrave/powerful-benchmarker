import glob
import os

import pandas as pd
import seaborn as sns


def save_plot(output_folder, df):
    Ns = [100, 10, 1]
    x_axis_title = "Standard deviation of noise added to target-domain accuracy"
    keys = [f"Top {N} Accuracy" for N in Ns] + ["Noise Standard Deviation"]
    new_keys = [f"Avg Accuracy of Top {N}" for N in Ns] + [x_axis_title]
    df = df.rename(columns={k: v for k, v in zip(keys, new_keys)})

    df = pd.melt(
        df,
        id_vars=[x_axis_title],
        value_vars=[
            "Weighted Spearman Correlation",
            *new_keys,
        ],
        var_name="Metric",
        value_name="Correlation with original data",
    )

    sns.set(font_scale=1, style="whitegrid", rc={"figure.figsize": (8, 8)})
    plot = sns.lineplot(
        data=df,
        x=x_axis_title,
        y="Correlation with original data",
        hue="Metric",
    )
    sns.move_legend(plot, "lower left")
    fig = plot.get_figure()
    fig.savefig(
        os.path.join(output_folder, "resilience.png"),
        bbox_inches="tight",
    )
    fig.clf()


def main():
    folder = "plots/resilience_to_noise"
    folders = glob.glob(os.path.join(folder, "*"))

    all_dfs = []
    for f in folders:
        print("plotting", f)
        filename = os.path.join(f, "df.pkl")
        if os.path.isfile(filename):
            df = pd.read_pickle(filename)
            all_dfs.append(df)
            # save_plot(f, df)

    df = pd.concat(all_dfs, axis=0).reset_index()
    save_plot(folder, df)


main()
