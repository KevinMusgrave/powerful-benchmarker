import pandas as pd

from .df_utils import drop_validator_cols, exp_specific_columns


def add_IM(df):
    e = df[df["validator"] == "Entropy"]
    d = df[df["validator"] == "Diversity"]

    e = drop_validator_cols(e).rename(columns={"score": "entropy_score"})
    d = drop_validator_cols(d).rename(columns={"score": "diversity_score"})
    im = e.merge(d, on=exp_specific_columns(e, ["entropy_score", "diversity_score"]))

    im["score"] = im["entropy_score"] + im["diversity_score"]
    im["validator"] = "IM"
    im["validator_args"] = "{}"
    im = im.drop(columns=["entropy_score", "diversity_score"])

    return pd.concat([df, im], axis=0)
