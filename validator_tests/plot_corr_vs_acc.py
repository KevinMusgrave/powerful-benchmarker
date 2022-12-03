import argparse
import os
import sys

import pandas as pd
import seaborn as sns

sys.path.insert(0, ".")

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


def save_boxplot(folder_name, df, x, y, filename, figsize=(4.8, 4.8)):
    sns.set(style="whitegrid", rc={"figure.figsize": figsize})
    plot = sns.boxplot(data=df, x=x, y=y)
    fig = plot.get_figure()
    fig.savefig(
        os.path.join(folder_name, f"{filename}.png"),
        bbox_inches="tight",
    )
    fig.clf()


def plot_globally_ranked(folder_name, one_adapter, corr_name, adapter_name):
    groupby = group_by_task_validator(per_adapter=True)
    ranks = one_adapter.groupby(groupby)["score"].rank(method="min", ascending=False)
    one_adapter["rank"] = ranks
    scatter_plot(
        folder_name,
        df=one_adapter,
        x=corr_name,
        y=TARGET_ACCURACY,
        filename=f"{adapter_name}_all",
        c="rank",
        figsize=(20, 20),
    )
    scatter_plot(
        folder_name,
        df=one_adapter[one_adapter["rank"] < 200],
        x=corr_name,
        y=TARGET_ACCURACY,
        filename=f"{adapter_name}_top200",
        figsize=(4.8, 4.8),
        s=1,
    )
    return one_adapter


def plot_trial_ranked(folder_name, one_adapter, corr_name, adapter_name):
    one_adapter = _get_best_accuracy_per_adapter(
        one_adapter, nlargest=100, rank_by="score", return_ranks=True
    )
    scatter_plot(
        folder_name,
        df=one_adapter,
        x=corr_name,
        y=TARGET_ACCURACY,
        filename=f"{adapter_name}_per_trial",
        c="rank",
        figsize=(20, 20),
        s=1,
    )
    scatter_plot(
        folder_name,
        df=one_adapter[one_adapter["rank"] < 5],
        x=corr_name,
        y=TARGET_ACCURACY,
        filename=f"{adapter_name}_per_trial_top5",
        figsize=(4.8, 4.8),
        s=1,
    )
    return one_adapter


def plot_best_pairs(folder_name, df, corr_name):
    df = unify_validator_columns(
        df, new_col_name="unified_validator", drop_validator_args=False
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
        mask |= (df["adapter"] == adapter) & (df["unified_validator"] == validator)
    df = df[mask]
    gdf = plot_globally_ranked(folder_name, df, corr_name, "ALL")
    tdf = plot_trial_ranked(folder_name, df, corr_name, "ALL")
    gdf = gdf.sort_values(by=["adapter"])
    tdf = tdf.sort_values(by=["adapter"])

    save_boxplot(
        folder_name,
        gdf[gdf["rank"] < 200],
        "adapter",
        TARGET_ACCURACY,
        "AAA_best_pairs_global_boxplot",
    )
    save_boxplot(
        folder_name,
        tdf[tdf["rank"] < 5],
        "adapter",
        TARGET_ACCURACY,
        "AAA_best_pairs_per_trial_boxplot",
    )


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


def get_fn(args):
    def fn(output_folder, df):
        corr_name = "weighted_spearman"
        corr = _get_correlation(df.copy(), True, 0.0, corr_name)
        corr = pd.melt(
            corr,
            id_vars=["validator", "validator_args", "task"],
            value_vars=adapter_names(),
            var_name="adapter",
            value_name=corr_name,
        )
        assert len(df["task"].unique()) == 1
        folder_name = get_folder_name(output_folder, df)

        plot_best_pairs(folder_name, corr.merge(df), corr_name)

        # for a in corr["adapter"].unique():
        #     print(a)
        #     one_adapter = corr[corr["adapter"] == a]
        #     one_adapter = one_adapter.merge(df)
        #     plot_globally_ranked(folder_name, one_adapter, corr_name, a)
        #     plot_trial_ranked(folder_name, one_adapter, corr_name, a)

    return fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--output_folder", type=str, default="plots/corr_vs_acc")
    create_main.add_main_args(parser)
    args = parser.parse_args()
    create_main.main(args, get_fn(args), get_fn(args))
