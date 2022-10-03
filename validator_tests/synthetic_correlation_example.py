import argparse
import os
import sys

import numpy as np
import seaborn as sns
from pytorch_adapt.utils import common_functions as c_f
from scipy.stats import spearmanr

sys.path.insert(0, ".")

from validator_tests.utils.weighted_spearman import weighted_spearman


def save_plot(x, y, plots_folder, filename):
    plot = sns.scatterplot(x=x, y=y, s=2)
    fig = plot.get_figure()
    c_f.makedir_if_not_there(plots_folder)
    fig.savefig(
        os.path.join(plots_folder, f"{filename}.png"),
        bbox_inches="tight",
    )
    fig.clf()


def step_example():
    num_samples = 10000
    x = np.arange(num_samples) / num_samples
    y = np.arange(num_samples) / num_samples
    y_bad = np.concatenate([y[:9000], np.arange(1000) / num_samples], axis=0)
    y_good = y
    return x, y_bad, y_good


def noise_example():
    num_samples = 10000
    x = np.arange(num_samples) / num_samples
    y = np.arange(num_samples) / num_samples
    noise = np.random.randn(num_samples)
    y_bad = y - noise * x * 0.5
    y_good = y - noise * (x[::-1]) * 0.5
    return x, y_bad, y_good


def main(args):
    for name, (x, y_bad, y_good) in [
        ("step", step_example()),
        ("noise", noise_example()),
    ]:

        print("\n\n", name)
        print("spearman", spearmanr(x, y_bad).correlation)
        print("weighted spearman", weighted_spearman(x, y_bad, 2))
        print("spearman", spearmanr(x, y_good).correlation)
        print("weighted spearman", weighted_spearman(x, y_good, 2))

        save_plot(
            x,
            y_bad,
            os.path.join(args.output_folder, "synthetic_correlation_examples"),
            f"{name}_bad",
        )
        save_plot(
            x,
            y_good,
            os.path.join(args.output_folder, "synthetic_correlation_examples"),
            f"{name}_good",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output_folder", type=str, default="plots")
    args = parser.parse_args()
    main(args)
