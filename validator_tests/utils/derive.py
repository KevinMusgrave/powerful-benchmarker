import pandas as pd

from .df_utils import (
    drop_validator_cols,
    exp_specific_columns,
    remove_arg_from_validator_args,
)


def add_derived_scores(df):
    for x in [add_IM, add_NegSND, add_BNMFull]:
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


def add_BNMFull(df):
    x = df[df["validator"] == "BNM"]
    src = x[x["validator_args"].str.contains('"split": "src_train"')]
    target = x[x["validator_args"].str.contains('"split": "target_train"')]

    src = drop_validator_cols(src, drop_validator_args=False).rename(
        columns={"score": "src_BNM_score"}
    )
    target = drop_validator_cols(target, drop_validator_args=False).rename(
        columns={"score": "target_BNM_score"}
    )

    src = remove_arg_from_validator_args(src, ["split"])
    target = remove_arg_from_validator_args(target, ["split"])

    bnm_full = src.merge(
        target,
        on=exp_specific_columns(src, exclude=["src_BNM_score", "target_BNM_score"]),
    )
    bnm_full = bnm_full.assign(
        score=bnm_full["src_BNM_score"] + bnm_full["target_BNM_score"],
        validator="BNMFull",
    )
    bnm_full = bnm_full.drop(columns=["src_BNM_score", "target_BNM_score"])
    return pd.concat([df, bnm_full], axis=0)
