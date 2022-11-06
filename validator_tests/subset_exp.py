import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_adapt.utils import common_functions as c_f

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.eval_validators import eval_validators
from validator_tests.utils import create_main
from validator_tests.utils.constants import add_exp_group_args


def get_intervals():
    e = list(range(1, 11))
    r = list(range(1, 11))
    return e, r


def get_folder(output_folder, e, r):
    return os.path.join(output_folder, "subsets", f"e{e}_r{r}")


def metric_filenames():
    return [
        {
            "metric": "best_accuracy_per_adapter_5",
            "merge_on": ["adapter", "task"],
            "compare": ["target_train_micro"],
        },
        {
            "metric": "best_accuracy_per_adapter_ranked_by_score_5",
            "merge_on": ["adapter", "task", "validator", "validator_args"],
            "compare": ["target_train_micro"],
        },
        {
            "metric": "spearman_0.0_src_threshold",
            "merge_on": ["validator", "validator_args", "task"],
            "compare": ["spearman"],
        },
        {
            "metric": "weighted_spearman_0.0_src_threshold",
            "merge_on": ["validator", "validator_args", "task"],
            "compare": ["weighted_spearman"],
        },
    ]


def create_subsets(output_folder, df):
    epoch_intervals, run_intervals = get_intervals()
    for e in epoch_intervals:
        for r in run_intervals:
            print(e, r)
            curr_df = df[
                (df["epoch"].astype(int) % e == 0)
                & (df["trial_num"].astype(int) % r == 0)
            ]
            curr_folder = get_folder(output_folder, e, r)
            eval_validators(curr_folder, curr_df, [0.0], 5)


def eval_subsets(subsets_folder, original_folder):
    dataset_folders = glob.glob(os.path.join(original_folder, "*"))
    dataset_names = [os.path.basename(x) for x in dataset_folders]
    epoch_intervals, run_intervals = get_intervals()
    for d in dataset_names:
        curr_original_folder = os.path.join(original_folder, d)
        for m in metric_filenames():
            all_diffs = {"epoch_interval": [], "run_interval": [], "avg_diff": []}
            for e in epoch_intervals:
                for r in run_intervals:
                    print(d, m, e, r)
                    curr_folder = os.path.join(get_folder(subsets_folder, e, r), d)
                    if not os.path.isdir(curr_folder):
                        continue
                    original = pd.read_pickle(
                        os.path.join(curr_original_folder, f"{m['metric']}.pkl")
                    )
                    curr = pd.read_pickle(
                        os.path.join(curr_folder, f"{m['metric']}.pkl")
                    )
                    merged = original.merge(
                        curr, on=m["merge_on"], suffixes=("_full", "_subset")
                    )
                    curr_diffs = []
                    for c in m["compare"]:
                        curr_diffs.append(
                            np.mean(
                                merged[f"{c}_subset"].values
                                - merged[f"{c}_full"].values
                            )
                        )
                    all_diffs["epoch_interval"].append(e)
                    all_diffs["run_interval"].append(r)
                    all_diffs["avg_diff"].append(np.mean(curr_diffs))

            all_diffs = pd.DataFrame(all_diffs)
            diffs_folder = os.path.join(subsets_folder, "diffs", d)
            c_f.makedir_if_not_there(diffs_folder)
            filename = os.path.join(diffs_folder, f"{m['metric']}")
            all_diffs.to_csv(f"{filename}.csv", index=False)
            all_diffs.to_pickle(f"{filename}.pkl")


def plot_diffs(subsets_folder):
    folders = glob.glob(os.path.join(subsets_folder, "diffs", "*"))
    for fol in folders:
        files = glob.glob(os.path.join(fol, "*.pkl"))
        for fil in files:
            df = pd.read_pickle(fil)
            df = df.pivot(
                index=["epoch_interval"], columns=["run_interval"], values=["avg_diff"]
            )
            df = df.droplevel(0, axis=1)

            dataset_name = os.path.basename(fol)
            plot = sns.heatmap(data=df)
            fig = plot.get_figure()
            plots_folder = os.path.join(subsets_folder, "plots", dataset_name)
            c_f.makedir_if_not_there(plots_folder)
            fig.savefig(
                os.path.join(plots_folder, f"{Path(fil).stem}.png"),
                bbox_inches="tight",
            )
            fig.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--original_folder", type=str, default="tables")
    parser.add_argument("--output_folder", type=str, default="tables_subsets")
    parser.add_argument("--action", type=str, choices=["create", "eval", "plot"])
    parser.add_argument("--create_subsets", action="store_true")
    create_main.add_main_args(parser)
    args = parser.parse_args()

    if args.action == "create":
        create_main.main(args, create_subsets, create_subsets)
    elif args.action == "eval":
        eval_subsets(args.output_folder, args.original_folder)
    elif args.action == "plot":
        plot_diffs(args.output_folder)
