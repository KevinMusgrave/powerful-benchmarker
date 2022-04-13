import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils.constants import (
    PER_SRC_FILENAME,
    PER_SRC_PER_ADAPTER_FILENAME,
    PROCESSED_DF_FILENAME,
)
from validator_tests.utils.plot_heatmap import (
    plot_heatmap,
    plot_heatmap_average_across_adapters,
    plot_heatmap_per_adapter,
)
from validator_tests.utils.plot_val_vs_acc import plot_val_vs_acc


def the_top_ones(df, key, per_adapter=False):
    df = df[df["validator"] != "Accuracy"]
    group_by = ["validator", "validator_args", key]
    if per_adapter:
        group_by += ["adapter"]
    df = df[[*group_by, "src_threshold"]]
    df = df.groupby(group_by)["src_threshold"].min()
    df = df.reset_index(name="src_threshold")
    df = df.sort_values(by=[key], ascending=False)
    print(df.iloc[:20])


def get_processed_df(exp_folder):
    filename = os.path.join(exp_folder, PROCESSED_DF_FILENAME)
    return pd.read_pickle(filename)


def get_per_src_threshold_df(exp_folder, per_adapter):
    basename = PER_SRC_PER_ADAPTER_FILENAME if per_adapter else PER_SRC_FILENAME
    filename = os.path.join(exp_folder, basename)
    return pd.read_pickle(filename)


def main(args):
    exp_folder = os.path.join(args.exp_folder, args.exp_group)
    if not args.no_scatter:
        df = get_processed_df(exp_folder)
        plot_val_vs_acc(df, args.plots_folder, False, args.scatter_plot_validator_set)

    per_src = get_per_src_threshold_df(exp_folder, False)
    the_top_ones(per_src, "predicted_best_acc")
    the_top_ones(per_src, "correlation")
    plot_heatmap(per_src, args.plots_folder)

    per_src = get_per_src_threshold_df(exp_folder, True)
    the_top_ones(per_src, "predicted_best_acc", per_adapter=True)
    the_top_ones(per_src, "correlation", per_adapter=True)
    plot_heatmap_per_adapter(per_src, args.plots_folder)
    plot_heatmap_average_across_adapters(per_src, args.plots_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--exp_group", type=str, required=True)
    parser.add_argument("--plots_folder", type=str, default="plots")
    parser.add_argument(
        "--scatter_plot_validator_set", nargs="+", type=str, default=None
    )
    parser.add_argument("--no_scatter", action="store_true")
    args = parser.parse_args()
    main(args)
