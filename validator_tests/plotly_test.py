import argparse
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.eval_validators import group_by_task_validator
from validator_tests.utils import create_main
from validator_tests.utils.constants import TARGET_ACCURACY, add_exp_group_args
from validator_tests.utils.df_utils import unify_validator_columns


def get_best_accuracy_per_adapter(df):
    rank_by = "score"
    groupby = group_by_task_validator(per_adapter=True)
    groupby_with_trial_params = groupby + ["trial_params"]

    # best score per trial param
    ranked = df.groupby(groupby_with_trial_params)[rank_by].rank(
        method="min", ascending=False
    )
    to_save = df[ranked <= 1]

    # remove duplicate scores for a trial by taking the earliest epoch
    return to_save.sort_values(by=["epoch"]).drop_duplicates(
        subset=groupby_with_trial_params
    )


# https://stackoverflow.com/a/64146570
def add_dummy_validator_column(df):
    dfg = pd.DataFrame({"validator": df["validator"].unique()})
    dfg["validator_index"] = dfg.index
    return pd.merge(df, dfg, on="validator", how="left")


def create_plot(original_df, validator_name):
    true_max = original_df[TARGET_ACCURACY].max()
    true_min = original_df[TARGET_ACCURACY].min()
    df = original_df.copy()
    df = df[(df["adapter"] == "DANNConfig") & (df["validator"] == validator_name)]
    df = get_best_accuracy_per_adapter(df)
    df = unify_validator_columns(df)
    df = add_dummy_validator_column(df)

    applied_df = df.apply(
        lambda row: json.loads(row.trial_params), axis="columns", result_type="expand"
    )
    df = pd.concat([df, applied_df], axis="columns")

    trial_param_keys = json.loads(df["trial_params"].iloc[0]).keys()
    trial_param_dimensions = [
        dict(
            label="log_lr",
            values=np.log10(df[x].values),
            tickvals=np.linspace(
                np.log10(df[x].min()), np.log10(df[x].max()), 10, endpoint=True
            ),
        )
        if x == "lr"
        else dict(label=x, values=df[x])
        for x in trial_param_keys
    ]

    validator_dimension = dict(
        tickvals=df["validator_index"],
        ticktext=df["validator"],
        label="Validator",
        values=df["validator_index"],
    )

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df[TARGET_ACCURACY],
                colorscale="viridis",
                showscale=True,
                cmin=true_min,
                cmax=true_max,
            ),
            dimensions=list(
                [
                    validator_dimension,
                    *trial_param_dimensions,
                    dict(label="Feature Layer", values=df["feature_layer"]),
                ]
            ),
        )
    )
    fig.write_html(f"plotly_test_{validator_name}.html")


def create_subsets(output_folder, original_df):
    for validator_name in original_df["validator"].unique():
        print(validator_name)
        create_plot(original_df, validator_name)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    create_main.add_main_args(parser)
    args = parser.parse_args()
    create_main.main(args, create_subsets, create_subsets)
