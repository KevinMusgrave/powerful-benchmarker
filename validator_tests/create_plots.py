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
from validator_tests.utils.threshold_utils import (
    get_all_per_task_validator,
    get_all_per_task_validator_adapter,
    get_per_threshold,
)


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


def get_per_x_threshold(df, exp_folder, read_existing, nlargest, per_adapter=False):
    src_basename = PER_SRC_PER_ADAPTER_FILENAME if per_adapter else PER_SRC_FILENAME
    src_filename = os.path.join(exp_folder, src_basename)
    if read_existing and os.path.isfile(src_filename):
        per_src = pd.read_pickle(src_filename)
    else:
        fn = (
            get_all_per_task_validator_adapter(nlargest)
            if per_adapter
            else get_all_per_task_validator(nlargest)
        )
        per_src = get_per_threshold(df, fn)
        per_src.to_pickle(src_filename)
    return per_src


def main(args):
    exp_folder = os.path.join(args.exp_folder, args.exp_group)
    df = get_processed_df(exp_folder)
    if not args.no_scatter:
        plot_val_vs_acc(df, args.plots_folder, False, args.scatter_plot_validator_set)

    per_src = get_per_x_threshold(df, exp_folder, args.read_existing, nlargest=100)
    the_top_ones(per_src, "predicted_best_acc")
    the_top_ones(per_src, "correlation")
    plot_heatmap(per_src, args.plots_folder)

    per_src = get_per_x_threshold(
        df, exp_folder, args.read_existing, nlargest=10, per_adapter=True
    )
    the_top_ones(per_src, "predicted_best_acc", per_adapter=True)
    the_top_ones(per_src, "correlation", per_adapter=True)
    plot_heatmap_per_adapter(per_src, args.plots_folder)
    plot_heatmap_average_across_adapters(per_src, args.plots_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--exp_group", type=str, required=True)
    parser.add_argument("--plots_folder", type=str, default="plots")
    parser.add_argument("--read_existing", action="store_true")
    parser.add_argument(
        "--scatter_plot_validator_set", nargs="+", type=str, default=None
    )
    parser.add_argument("--no_scatter", action="store_true")
    args = parser.parse_args()
    main(args)
