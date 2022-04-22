import argparse
import os
import sys

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils
from validator_tests.utils.constants import (
    add_exp_group_args,
    get_exp_groups_with_matching_tasks,
    get_name_from_exp_groups,
    get_per_src_threshold_df,
    get_processed_df,
)
from validator_tests.utils.plot_heatmap import (
    plot_heatmap,
    plot_heatmap_average_across_adapters,
    plot_heatmap_per_adapter,
)
from validator_tests.utils.plot_val_vs_acc import plot_val_vs_acc


def scatter_and_heatmap(exp_folder, exp_groups, plots_folder, per_feature_layer, df):
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


def main(args):
    exp_groups = utils.get_exp_groups(args)
    for e in exp_groups:
        print("exp_group", e)
        # do per feature layer
        exp_folder = os.path.join(args.exp_folder, e)
        df = get_processed_df(exp_folder)
        if df is not None:
            scatter_and_heatmap(exp_folder, [e], args.plots_folder, True, df)

    # now do with feature layers in one dataframe
    # which are saved in args.exp_folder
    combined_dfs, combined_exp_groups = get_exp_groups_with_matching_tasks(
        args.exp_folder, exp_groups, return_dfs=True
    )
    for df, e in zip(combined_dfs, combined_exp_groups):
        print("exp_groups", e)
        scatter_and_heatmap(args.exp_folder, e, args.plots_folder, False, df)


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

    # topN here refers to the topN used in per_src_threshold.py
    parser.add_argument("--topN", type=int, default=200)
    parser.add_argument("--topN_per_adapter", type=int, default=20)
    args = parser.parse_args()
    main(args)
