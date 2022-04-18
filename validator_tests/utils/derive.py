import pandas as pd

from .df_utils import (
    drop_validator_cols,
    exp_specific_columns,
    remove_arg_from_validator_args,
)


def add_derived_scores(df):
    for x in [
        add_IM,
        add_NegSND,
        add_BNMSummed,
        add_BSPSummed,
        add_EntropySummed,
        add_DiversitySummed,
        add_IMSummed,
    ]:
        df = x(df)
    return df


def add_IM(df):
    e = df[df["validator"] == "Entropy"]
    d = df[df["validator"] == "Diversity"]

    if len(e) == 0 or len(d) == 0:
        return df

    e = drop_validator_cols(e, drop_validator_args=False).rename(
        columns={"score": "entropy_score"}
    )
    d = drop_validator_cols(d, drop_validator_args=False).rename(
        columns={"score": "diversity_score"}
    )
    im = e.merge(
        d, on=exp_specific_columns(e, exclude=["entropy_score", "diversity_score"])
    )

    im = im.assign(
        score=im["entropy_score"] + im["diversity_score"],
        validator="IM",
    )
    im = im.drop(columns=["entropy_score", "diversity_score"])

    return pd.concat([df, im], axis=0)


def add_NegSND(df):
    x = df[df["validator"] == "SND"]
    if len(x) == 0:
        return df
    x = drop_validator_cols(x, drop_validator_args=False).rename(
        columns={"score": "SND_score"}
    )
    x = x.assign(score=-x["SND_score"], validator="NegSND")
    x = x.drop(columns=["SND_score"])
    return pd.concat([df, x], axis=0)


def _add_src_and_target(df, validator_name):
    x = df[df["validator"] == validator_name]
    if len(x) == 0:
        return df
    src = x[x["validator_args"].str.contains('"split": "src_train"')]
    target = x[x["validator_args"].str.contains('"split": "target_train"')]

    src_score_name = f"src_{validator_name}_score"
    target_score_name = f"target_{validator_name}_score"

    src = drop_validator_cols(src, drop_validator_args=False).rename(
        columns={"score": src_score_name}
    )
    target = drop_validator_cols(target, drop_validator_args=False).rename(
        columns={"score": target_score_name}
    )

    src = remove_arg_from_validator_args(src, ["split"])
    target = remove_arg_from_validator_args(target, ["split"])

    summed = src.merge(
        target,
        on=exp_specific_columns(src, exclude=[src_score_name, target_score_name]),
    )
    summed = summed.assign(
        score=summed[src_score_name] + summed[target_score_name],
        validator=f"{validator_name}Summed",
    )
    summed = summed.drop(columns=[src_score_name, target_score_name])
    return pd.concat([df, summed], axis=0)


def add_BNMSummed(df):
    return _add_src_and_target(df, "BNM")


def add_BSPSummed(df):
    return _add_src_and_target(df, "BSP")


def add_EntropySummed(df):
    return _add_src_and_target(df, "Entropy")


def add_DiversitySummed(df):
    return _add_src_and_target(df, "Diversity")


def add_IMSummed(df):
    return _add_src_and_target(df, "IM")
