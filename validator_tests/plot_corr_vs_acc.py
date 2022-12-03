import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.eval_validators import _get_correlation, group_by_task_validator
from validator_tests.utils import create_main
from validator_tests.utils.constants import TARGET_ACCURACY, add_exp_group_args
from validator_tests.utils.df_utils import get_name_from_df
from validator_tests.utils.plot_val_vs_acc import scatter_plot


def get_folder_name(folder, full_df):
    return os.path.join(folder, get_name_from_df(full_df, assert_one_task=True))


def get_fn(args):
    def fn(output_folder, df):
        name = "weighted_spearman"
        corr = _get_correlation(df.copy(), True, 0.0, name)
        corr = pd.melt(
            corr,
            id_vars=["validator", "validator_args", "task"],
            value_vars=[
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
            ],
            var_name="adapter",
            value_name=name,
        )
        assert len(df["task"].unique()) == 1
        for a in corr["adapter"].unique():
            one_adapter = corr[corr["adapter"] == a]
            one_adapter = one_adapter.sort_values(by=[name], ascending=False)
            one_adapter = one_adapter.merge(df)
            groupby = group_by_task_validator(per_adapter=True)
            ranks = one_adapter.groupby(groupby)["score"].rank(
                method="min", ascending=False
            )
            one_adapter["rank"] = ranks
            print(one_adapter)

            folder_name = get_folder_name(output_folder, df)
            scatter_plot(
                folder_name,
                df=one_adapter,
                x=name,
                y=TARGET_ACCURACY,
                filename=a,
                c="rank",
                figsize=(20, 20),
            )

    return fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--output_folder", type=str, default="plots/corr_vs_acc")
    create_main.add_main_args(parser)
    args = parser.parse_args()
    create_main.main(args, get_fn(args), get_fn(args))
