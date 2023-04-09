import argparse
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_adapt.utils import common_functions as c_f
from scipy.stats import spearmanr

sys.path.insert(0, ".")

from validator_tests.utils.plot_val_vs_acc import scatter_plot
from validator_tests.utils.weighted_spearman import weighted_spearman


def save_plot(x, y, plots_folder, filename):
    df = pd.DataFrame({"Validation Score": x, "Target Accuracy": y})
    sns.set(font_scale=2, style="whitegrid", rc={"figure.figsize": (10, 10)})
    plot = sns.scatterplot(data=df, x="Validation Score", y="Target Accuracy", s=4)
    fig = plot.get_figure()
    c_f.makedir_if_not_there(plots_folder)
    fig.savefig(
        os.path.join(plots_folder, f"{filename}.png"),
        bbox_inches="tight",
    )
    fig.clf()


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def step_example():
    num_samples = 10000
    x = np.arange(num_samples) / num_samples
    y = np.arange(num_samples) / num_samples
    y_bad = np.concatenate([y[:9000], np.arange(1000) / num_samples], axis=0)
    y_good = y
    return x, normalize(y_bad), normalize(y_good)


def outliers_example1():
    num_samples = 10000
    x = np.arange(num_samples) / num_samples
    y = np.arange(num_samples) / num_samples
    y_bad = np.copy(y)
    y_bad[:1000] = np.random.randn(1000) + 10
    y_good = y
    return x, normalize(y_bad), normalize(y_good)


def outliers_example2():
    num_samples = 10000
    x = np.arange(num_samples) / num_samples
    y = np.arange(num_samples) / num_samples
    y_bad = np.copy(y)
    y_bad[-1000:] = np.random.randn(1000) - 10
    y_good = y
    return x, normalize(y_bad), normalize(y_good)


def noise_example():
    num_samples = 40000
    x = np.arange(num_samples) / num_samples
    y = np.arange(num_samples) * 2 / num_samples
    noise = np.random.randn(num_samples)
    y_bad = y - noise * x * 0.5
    y_good = y - noise * (x[::-1]) * 0.5
    return x, normalize(y_bad), normalize(y_good)


def main(args):
    for name, (x, y_bad, y_good) in [
        ("step", step_example()),
        ("noise", noise_example()),
        ("outliers1", outliers_example1()),
        ("outliers2", outliers_example2()),
    ]:

        print("\n\n", name)
        spearman_bad = spearmanr(x, y_bad).correlation
        weighted_spearman_bad = weighted_spearman(x, y_bad, 2)
        spearman_good = spearmanr(x, y_good).correlation
        weighted_spearman_good = weighted_spearman(x, y_good, 2)

        print("spearman_bad", spearman_bad)
        print("weighted_spearman_bad", weighted_spearman_bad)
        print("spearman_good", spearman_good)
        print("weighted_spearman_good", weighted_spearman_good)

        plots_folder = os.path.join(
            args.output_folder, "synthetic_correlation_examples"
        )
        x_label = "Validation Score"
        y_label = "Target Accuracy"
        kwargs = {
            "x_label": x_label,
            "y_label": y_label,
            "font_scale": 2,
            "figsize": (10, 10),
            "s": 0.2,
        }

        scatter_plot(
            plots_folder,
            pd.DataFrame({x_label: x, y_label: y_bad}),
            x_label,
            y_label,
            f"{name}_bad",
            **kwargs,
        )

        scatter_plot(
            plots_folder,
            pd.DataFrame({x_label: x, y_label: y_good}),
            x_label,
            y_label,
            f"{name}_good",
            **kwargs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output_folder", type=str, default="plots")
    args = parser.parse_args()
    main(args)
