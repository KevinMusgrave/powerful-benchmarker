import argparse
import os
import sys

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import create_main
from validator_tests.utils.constants import (
    add_exp_group_args,
    get_name_from_exp_groups,
    get_per_src_threshold_df,
)
from validator_tests.utils.plot_heatmap import (
    plot_heatmap,
    plot_heatmap_average_across_adapters,
    plot_heatmap_per_adapter,
)
from validator_tests.utils.plot_val_vs_acc import plot_val_vs_acc


def scatter_and_heatmap(exp_folder, exp_groups, plots_folder, df, per_feature_layer):
    if args.scatter:
        plot_val_vs_acc(
            df,
            plots_folder,
            per_adapter=False,
            per_feature_layer=per_feature_layer,
            validator_set=args.scatter_plot_validator_set,
        )
    if args.heatmap:
        plots_folder = os.path.join(plots_folder, get_name_from_exp_groups(exp_groups))
        per_src = get_per_src_threshold_df(exp_folder, False, args.topN, exp_groups)
        plot_heatmap(per_src, plots_folder, args.topN)

        per_src = get_per_src_threshold_df(
            exp_folder, True, args.topN_per_adapter, exp_groups
        )
        plot_heatmap_per_adapter(per_src, plots_folder, args.topN_per_adapter)
        plot_heatmap_average_across_adapters(
            per_src, plots_folder, args.topN_per_adapter
        )


def fn1(*args):
    scatter_and_heatmap(*args, per_feature_layer=True)


def fn2(*args):
    scatter_and_heatmap(*args, per_feature_layer=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    # topN here refers to the topN used in per_src_threshold.py
    create_main.add_topN_args(parser)
    parser.add_argument("--output_folder", type=str, default="plots")
    parser.add_argument(
        "--scatter_plot_validator_set", nargs="+", type=str, default=None
    )
    parser.add_argument("--scatter", action="store_true")
    parser.add_argument("--heatmap", action="store_true")
    args = parser.parse_args()
    create_main.main(args, fn1, fn2)
