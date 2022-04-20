import argparse
import os
import sys

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils
from validator_tests.utils.constants import (
    add_exp_group_args,
    get_per_src_threshold_df,
    get_processed_df,
)
from validator_tests.utils.plot_heatmap import (
    plot_heatmap,
    plot_heatmap_average_across_adapters,
    plot_heatmap_per_adapter,
)
from validator_tests.utils.plot_val_vs_acc import plot_val_vs_acc


def scatter_and_heatmap(args, exp_group):
    exp_folder = os.path.join(args.exp_folder, exp_group)
    if args.scatter:
        df = get_processed_df(exp_folder)
        if df is not None:
            plot_val_vs_acc(
                df, args.plots_folder, False, args.scatter_plot_validator_set
            )

    if args.heatmap:
        per_src = get_per_src_threshold_df(exp_folder, False)
        plot_heatmap(per_src, args.plots_folder)

        per_src = get_per_src_threshold_df(exp_folder, True)
        plot_heatmap_per_adapter(per_src, args.plots_folder)
        plot_heatmap_average_across_adapters(per_src, args.plots_folder)


def main(args):
    exp_groups = utils.get_exp_groups(args)
    for e in exp_groups:
        scatter_and_heatmap(args, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--plots_folder", type=str, default="plots")
    parser.add_argument(
        "--scatter_plot_validator_set", nargs="+", type=str, default=None
    )
    parser.add_argument("--scatter", action="store_true")
    parser.add_argument("--heatmap", action="store_true")
    args = parser.parse_args()
    main(args)
