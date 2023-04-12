import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from pytorch_adapt.utils import common_functions as c_f
from scipy.stats import spearmanr

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.eval_validators import _get_correlation, group_by_task_validator
from validator_tests.plot_ranks_vs_acc import get_global_ranks
from validator_tests.utils import create_main
from validator_tests.utils.constants import TARGET_ACCURACY, add_exp_group_args
from validator_tests.utils.df_utils import get_name_from_df


def add_noise(original_df, scale):
    df = original_df.copy()
    df.groupby(["adapter", "trial_num", "epoch"]).size().reset_index().rename(
        columns={0: "count"}
    )
    df = df.rename(columns={"count": "noise"})
    df["noise"] = np.random.normal(scale=scale, size=(len(df)))
    df = original_df.merge(df)
    df[TARGET_ACCURACY] = df[TARGET_ACCURACY] + df["noise"]
    df[TARGET_ACCURACY] = df[TARGET_ACCURACY].clip(lower=0, upper=1)
    return df


def get_correlation(df, per_adapter, name="weighted_spearman"):
    return _get_correlation(
        df.copy(), per_adapter=per_adapter, src_threshold=0.0, name=name
    )


def get_acc(df, per_adapter, N):
    df = get_global_ranks(df, rank_by="score", per_adapter=per_adapter)
    df = df[df["rank"] <= N]
    df[f"top_{N}_acc"] = df.groupby(group_by_task_validator(per_adapter))[
        TARGET_ACCURACY
    ].transform("mean")
    keep = ["validator", "validator_args", f"top_{N}_acc"]
    if per_adapter:
        keep += ["adapter"]
    return df[keep].drop_duplicates()


def save_df(output_folder, df):
    accs = {}
    corr = get_correlation(df, per_adapter=False, name="spearman")
    wcorr = get_correlation(df, per_adapter=False)
    Ns = [1, 5, 10, 50, 100, 500, 1000, 5000]
    for N in Ns:
        accs[N] = get_acc(df, per_adapter=False, N=N)

    s = defaultdict(list)
    for scale in np.linspace(0, 0.2, 21):
        s["Noise Standard Deviation"].append(scale)
        df_with_noise = add_noise(df, scale)
        corr_with_noise = get_correlation(
            df_with_noise, per_adapter=False, name="spearman"
        )
        wcorr_with_noise = get_correlation(df_with_noise, per_adapter=False)
        s["Spearman Correlation"].append(
            spearmanr(
                corr_with_noise["spearman"].values,
                corr["spearman"].values,
            ).correlation
        )
        s["Weighted Spearman Correlation"].append(
            spearmanr(
                wcorr_with_noise["weighted_spearman"].values,
                wcorr["weighted_spearman"].values,
            ).correlation
        )
        for N in Ns:
            acc_with_noise = get_acc(df_with_noise, per_adapter=False, N=N)
            s[f"Top {N} Accuracy"].append(
                spearmanr(
                    acc_with_noise[f"top_{N}_acc"].values,
                    accs[N][f"top_{N}_acc"].values,
                ).correlation
            )

        print(s)

    sdf = pd.DataFrame.from_dict(s)

    output_folder = os.path.join(
        output_folder, get_name_from_df(df, assert_one_task=True)
    )
    c_f.makedir_if_not_there(output_folder)
    sdf.to_pickle(os.path.join(output_folder, "df.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument(
        "--output_folder", type=str, default="plots/resilience_to_noise"
    )
    create_main.add_main_args(parser)
    args = parser.parse_args()
    create_main.main(args, save_df, save_df)
