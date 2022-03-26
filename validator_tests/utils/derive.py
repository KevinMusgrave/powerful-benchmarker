import pandas as pd

from .df_utils import drop_validator_cols, exp_specific_columns


def add_derived_scores(df):
    for x in [add_IM, add_NegSND]:
        df = x(df)
    return df


def add_IM(df):
    e = df[df["validator"] == "Entropy"]
    d = df[df["validator"] == "Diversity"]

    e = drop_validator_cols(e).rename(columns={"score": "entropy_score"})
    d = drop_validator_cols(d).rename(columns={"score": "diversity_score"})
    im = e.merge(d, on=exp_specific_columns(e, ["entropy_score", "diversity_score"]))

    im = im.assign(
        score=im["entropy_score"] + im["diversity_score"],
        validator="IM",
        validator_args="{}",
    )
    im = im.drop(columns=["entropy_score", "diversity_score"])

    return pd.concat([df, im], axis=0)


def add_NegSND(df):
    x = df[df["validator"] == "SND"]
    x = drop_validator_cols(x, drop_validator_args=False).rename(
        columns={"score": "SND_score"}
    )
    x = x.assign(score=-x["SND_score"], validator="NegSND")
    x = x.drop(columns=["SND_score"])
    return pd.concat([df, x], axis=0)
