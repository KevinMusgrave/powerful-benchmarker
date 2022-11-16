import argparse
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import create_main
from validator_tests.utils.constants import TARGET_ACCURACY, add_exp_group_args


def create_subsets(output_folder, df):
    df = df[df["adapter"] == "DANNConfig"]
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

    print(trial_param_dimensions)

    print("len(df)", len(df))

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=df[TARGET_ACCURACY], colorscale="Electric", showscale=True),
            dimensions=list(
                [
                    *trial_param_dimensions,
                    dict(label="Feature Layer", values=df["feature_layer"]),
                ]
            ),
        )
    )
    fig.write_html("plotly_test.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    create_main.add_main_args(parser)
    args = parser.parse_args()
    create_main.main(args, create_subsets, create_subsets)
