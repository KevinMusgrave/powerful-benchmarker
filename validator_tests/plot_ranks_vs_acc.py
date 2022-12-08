import argparse
import os
import sys

import pandas as pd
import seaborn as sns

sys.path.insert(0, ".")

from pytorch_adapt.utils import common_functions as c_f

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.eval_validators import (
    _get_best_accuracy_per_adapter,
    _get_correlation,
    group_by_task_validator,
)
from validator_tests.utils import create_main
from validator_tests.utils.constants import TARGET_ACCURACY, add_exp_group_args
from validator_tests.utils.df_utils import get_name_from_df, unify_validator_columns
from validator_tests.utils.plot_val_vs_acc import scatter_plot


def get_folder_name(folder, full_df):
    return os.path.join(folder, get_name_from_df(full_df, assert_one_task=True))


def adapter_names():
    return [
        "ATDOCConfig",
        "BNMConfig",
        "BSPConfig",
        "CDANConfig",
        "DANNConfig",
        "GVBConfig",
        "IMConfig",
        "MCCConfig",
        "MCDConfig",
        "MMDConfig",
    ]


def get_global_ranks(df, rank_by):
    groupby = group_by_task_validator(per_adapter=True)
    corr_with_ranks = df.copy()
    ranks = corr_with_ranks.groupby(groupby)[rank_by].rank(
        method="min", ascending=False
    )
    corr_with_ranks["rank"] = ranks
    return corr_with_ranks


def plot_corr_vs_acc(df, max_rank, corr_name, folder, filename):
    to_plot = df[df["rank"] <= max_rank]
    to_plot = to_plot.sort_values(by=["adapter"])
    to_plot["adapter"] = to_plot["adapter"].str.replace("Config", "")
    sns.set(style="whitegrid", rc={"figure.figsize": (8, 8)})
    plot = sns.scatterplot(
        data=to_plot, x=corr_name, y=TARGET_ACCURACY, hue="rank_type", alpha=0.5
    )
    sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
    fig = plot.get_figure()
    c_f.makedir_if_not_there(folder)
    fig.savefig(
        os.path.join(folder, f"{filename}.png"),
        bbox_inches="tight",
    )
    fig.clf()


def plot_corr_vs_true_and_predicted(best, max_rank, corr_name, output_folder, filename):
    plot_corr_vs_acc(best, max_rank, corr_name, output_folder, filename)


def get_corr_df(df, corr_name):
    corr = _get_correlation(df.copy(), True, 0.0, corr_name)
    corr = pd.melt(
        corr,
        id_vars=["validator", "validator_args", "task"],
        value_vars=adapter_names(),
        var_name="adapter",
        value_name=corr_name,
    )
    assert len(corr["task"].unique()) == 1
    corr = corr.merge(df)
    corr = unify_validator_columns(
        corr, new_col_name="unified_validator", drop_validator_args=False
    )
    best_validators = [
        (
            x,
            "BNMSummedSrcVal_layer_logits"
            if x == "ATDOCConfig"
            else "Accuracy_average_micro_split_src_val",
        )
        for x in adapter_names()
    ]
    mask = False
    for adapter, validator in best_validators:
        mask |= (corr["adapter"] == adapter) & (corr["unified_validator"] == validator)
    corr_best_validators = corr[mask]

    return corr, corr_best_validators


def main_fn(output_folder, df, nlargest, nlargest_global):
    corr_name = "weighted_spearman"
    corr, corr_best_validators = get_corr_df(df, corr_name)

    best_by_score = _get_best_accuracy_per_adapter(
        corr_best_validators.copy(), nlargest=100000, rank_by="score", return_ranks=True
    )
    best_by_acc = _get_best_accuracy_per_adapter(
        corr_best_validators.copy(),
        nlargest=100000,
        rank_by=TARGET_ACCURACY,
        return_ranks=True,
    )
    best_by_score["rank_type"] = "by validation score"
    best_by_acc["rank_type"] = "by accuracy"
    best = pd.concat([best_by_score, best_by_acc], axis=0)
    plot_corr_vs_true_and_predicted(
        best, nlargest, corr_name, output_folder, f"selected_models_local_{nlargest}"
    )

    best_by_score = get_global_ranks(corr_best_validators.copy(), "score")
    best_by_acc = get_global_ranks(corr_best_validators.copy(), TARGET_ACCURACY)
    best_by_score["rank_type"] = "by validation score"
    best_by_acc["rank_type"] = "by accuracy"
    best = pd.concat([best_by_score, best_by_acc], axis=0)
    plot_corr_vs_true_and_predicted(
        best,
        nlargest_global,
        corr_name,
        output_folder,
        f"selected_models_global_{nlargest_global}",
    )

    for a in corr["adapter"].unique():
        one_adapter = corr[corr["adapter"] == a]
        best_by_score = get_global_ranks(one_adapter, "score")
        scatter_plot(
            output_folder,
            df=best_by_score,
            x=corr_name,
            y=TARGET_ACCURACY,
            filename=f"global_rank_heatmap_{a}",
            c="rank",
        )


def get_fn(args):
    def fn(output_folder, df):
        return main_fn(output_folder, df, args.nlargest, args.nlargest_global)

    return fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--output_folder", type=str, default="plots/ranks_vs_acc")
    parser.add_argument("--nlargest", type=int, default=5)
    parser.add_argument("--nlargest_global", type=int, default=200)
    create_main.add_main_args(parser)
    args = parser.parse_args()
    create_main.main(args, get_fn(args), get_fn(args))
