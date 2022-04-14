import argparse
import os
import sys

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils.constants import get_per_src_threshold_df, get_processed_df
from validator_tests.utils.plot_heatmap import (
    plot_heatmap,
    plot_heatmap_average_across_adapters,
    plot_heatmap_per_adapter,
)
from validator_tests.utils.plot_val_vs_acc import plot_val_vs_acc


def main(args):
    exp_folder = os.path.join(args.exp_folder, args.exp_group)
    if not args.no_scatter:
        df = get_processed_df(exp_folder)
        plot_val_vs_acc(df, args.plots_folder, False, args.scatter_plot_validator_set)

    per_src = get_per_src_threshold_df(exp_folder, False)
    plot_heatmap(per_src, args.plots_folder)

    per_src = get_per_src_threshold_df(exp_folder, True)
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
