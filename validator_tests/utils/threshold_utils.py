import numpy as np
import pandas as pd
import tqdm

from powerful_benchmarker.utils.score_utils import (
    pretrained_src_accuracy,
    pretrained_target_accuracy,
)

from .constants import EXPECTED_NUMBER_OF_CHECKPOINTS, TARGET_ACCURACY
from .df_utils import get_sorted_unique


def filter_by_acc(df, min_acc, domain_type):
    if domain_type == "src":
        split = "val"
    elif domain_type == "target":
        split = "train"
    return df[df[f"{domain_type}_{split}_micro"] >= min_acc]


def domain_type_str(domain_type):
    return "_".join(domain_type)


def per_threshold(df, pretrained_acc, domain_type, fn):
    print("pretrained_acc", pretrained_acc)
    all_df = []
    upper_bound = 2
    thresholds = np.linspace(0, upper_bound, 201)
    for threshold in tqdm.tqdm(thresholds):
        min_acc = [pa * threshold for pa in pretrained_acc]
        curr_df = df
        for i, k in enumerate(domain_type):
            curr_df = filter_by_acc(curr_df, np.round(min_acc[i], 4), k)
        if len(curr_df) == 0:
            continue
        if threshold == 0:
            assert len(curr_df) == len(df)
        t_str = f"{domain_type_str(domain_type)}_threshold"
        curr_df = fn(curr_df)
        curr_df = curr_df.assign(**{t_str: threshold})
        curr_df = curr_df.round({t_str: 2})
        all_df.append(curr_df)
    return pd.concat(all_df, axis=0, ignore_index=True)


def get_per_src_threshold(df, dataset, src_domain, fn):
    pretrained_acc = pretrained_src_accuracy(dataset, src_domain, "val", "micro")
    return per_threshold(df, [pretrained_acc], ["src"], fn)


def get_per_target_threshold(df, dataset, source_domain, target_domain, fn):
    pretrained_acc = pretrained_target_accuracy(
        dataset, source_domain, target_domain, "train", "micro"
    )
    return per_threshold(df, [pretrained_acc], ["target"], fn)


def assign_original_df_info(all_per_src, df):
    task = get_sorted_unique(df, "task", assert_one=True)[0]
    dataset = get_sorted_unique(df, "dataset", assert_one=True)[0]
    src_domains = get_sorted_unique(df, "src_domains", assert_one=True)[0]
    target_domains = get_sorted_unique(df, "target_domains", assert_one=True)[0]
    feature_layer = get_sorted_unique(df, "feature_layer")
    optimizer = get_sorted_unique(df, "optimizer")
    lr_multiplier = get_sorted_unique(df, "lr_multiplier")

    return all_per_src.assign(
        task=task,
        dataset=dataset,
        src_domains=[src_domains] * len(all_per_src),
        target_domains=[target_domains] * len(all_per_src),
        feature_layer=[feature_layer] * len(all_per_src),
        optimizer=[optimizer] * len(all_per_src),
        lr_multiplier=[lr_multiplier] * len(all_per_src),
    )


def get_per_threshold(df, fn):
    all_per_src = []
    for dataset in df["dataset"].unique():
        for src_domains in df["src_domains"].unique():
            for target_domains in df["target_domains"].unique():
                curr_df = df[
                    (df["dataset"] == dataset)
                    & (df["src_domains"] == src_domains)
                    & (df["target_domains"] == target_domains)
                ]
                per_src = get_per_src_threshold(curr_df, dataset, src_domains, fn)
                all_per_src.append(per_src)
    all_per_src = pd.concat(all_per_src, axis=0, ignore_index=True)
    return assign_original_df_info(all_per_src, df)


def get_corr(group_by):
    def fn(df):
        curr_df = (
            df.groupby(group_by)[["score", TARGET_ACCURACY]]
            .corr(method="spearman")
            .iloc[0::2, -1]
        )
        return (
            curr_df.reset_index()
            .drop([f"level_{len(group_by)}"], axis=1)
            .rename(columns={TARGET_ACCURACY: "correlation"})
            .dropna()
        )

    return fn


def get_predicted_best_acc(group_by, nlargest):
    def fn(df):
        return get_avg_top_n_acc_by_group(
            df, group_by, nlargest, "score", "predicted_best_acc_raw"
        )

    return fn


def get_all(group_by, nlargest):
    corr_fn = get_corr(group_by)
    acc_fn = get_predicted_best_acc(group_by, nlargest)

    def fn(df):
        df1 = corr_fn(df)
        df2 = acc_fn(df)
        num_past_threshold = (
            df.groupby(group_by).size().reset_index(name="num_past_threshold")
        )
        df = df1.merge(df2, on=group_by)
        return df.merge(num_past_threshold, on=group_by)

    return fn


def group_by_task(per_adapter):
    output = ["dataset", "src_domains", "target_domains"]
    if per_adapter:
        output.append("adapter")
    return output


def group_by_task_validator(per_adapter):
    return ["validator", "validator_args"] + group_by_task(per_adapter)


def get_all_per_task_validator(nlargest):
    return get_all(group_by_task_validator(per_adapter=False), nlargest)


def get_all_per_task_validator_adapter(nlargest):
    return get_all(group_by_task_validator(per_adapter=True), nlargest)


def get_avg_top_n_acc_by_group(df, group_by, nlargest, sort_by, new_col_name):
    ranked = df.groupby(group_by)[sort_by].rank(method="min", ascending=False)
    df = df[ranked <= nlargest]
    return (
        df.groupby(group_by)[TARGET_ACCURACY]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": new_col_name, "std": f"{new_col_name}_std"})
    )


def convert_predicted_best_acc_to_rel(df, per_x, per_adapter, nlargest, num_exp_groups):
    # the accuracy columns are duplicated for each validator/validator_args
    df = df.drop(columns=["validator", "validator_args", "score"]).drop_duplicates()
    max_num_checkpoints = EXPECTED_NUMBER_OF_CHECKPOINTS * num_exp_groups
    # TODO change this to != when experiments are done
    if len(df) > max_num_checkpoints:
        print(len(df))
        raise ValueError

    if per_adapter:
        max_num_checkpoints /= len(df["adapter"].unique())
    if per_x["num_past_threshold"].max() > max_num_checkpoints:
        print(per_x["num_past_threshold"].max())
        raise ValueError

    group_by = group_by_task(per_adapter=per_adapter)
    best_acc = get_avg_top_n_acc_by_group(
        df, group_by, nlargest, TARGET_ACCURACY, "best_acc"
    )
    per_x = per_x.merge(best_acc, on=group_by)
    per_x["predicted_best_acc"] = per_x["predicted_best_acc_raw"] / per_x["best_acc"]

    # https://math.stackexchange.com/a/2793257
    per_x["predicted_best_acc_std"] = (
        per_x["predicted_best_acc_raw_std"] / per_x["best_acc"]
    )

    # The rows with num_past_threshold < nlargest can have predicted_best_acc > 1
    # because they are unfairly focusing on a smaller subset.
    # So this check only applies when num_past_threshold >= nlargest
    strict_rows = per_x[per_x["num_past_threshold"] >= nlargest]
    if strict_rows["predicted_best_acc"].max() > (1 + 1e-8):
        print(strict_rows.loc[strict_rows["predicted_best_acc"].idxmax()])
        raise ValueError
    return per_x
