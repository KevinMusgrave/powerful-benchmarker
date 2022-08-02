import argparse
import os
import sys

from pytorch_adapt.utils import common_functions as c_f

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import create_main
from validator_tests.utils.constants import TARGET_ACCURACY, add_exp_group_args
from validator_tests.utils.df_utils import (
    add_task_column,
    get_name_from_df,
    get_sorted_unique,
)
from validator_tests.utils.weighted_spearman import spearman, weighted_spearman


def save_df(folder, full_df, df, filename):
    folder = os.path.join(folder, get_name_from_df(full_df, assert_one_task=True))
    c_f.makedir_if_not_there(folder)
    filename = os.path.join(folder, filename)
    df.to_csv(f"{filename}.csv", index=False)
    df.to_pickle(f"{filename}.pkl")


def assign_original_df_info(new_df, df):
    task = get_sorted_unique(df, "task", assert_one=True)[0]
    feature_layer = get_sorted_unique(df, "feature_layer")
    optimizer = get_sorted_unique(df, "optimizer")
    lr_multiplier = get_sorted_unique(df, "lr_multiplier")

    return new_df.assign(
        task=task,
        feature_layer=[feature_layer] * len(new_df),
        optimizer=[optimizer] * len(new_df),
        lr_multiplier=[lr_multiplier] * len(new_df),
    )


def group_by_task(per_adapter):
    output = ["dataset", "src_domains", "target_domains"]
    if per_adapter:
        output.append("adapter")
    return output


def group_by_task_validator(per_adapter):
    return ["validator", "validator_args"] + group_by_task(per_adapter)


def get_correlation(output_folder, df, per_adapter, src_threshold, score_fn, name):
    if src_threshold != 0:
        raise ValueError("src_threshold is temporarily disabled")
    # df = threshold_utils.filter_by_src_threshold(
    #     df, src_threshold, filter_action="set_to_nan"
    # )
    new_df = df.groupby(group_by_task_validator(per_adapter))[
        [TARGET_ACCURACY, "score"]
    ].apply(score_fn)
    new_df = new_df.reset_index(name=name)
    df = assign_original_df_info(new_df, df)

    filename = f"{name}_{src_threshold}_src_threshold"
    keep = ["validator", "validator_args", "task", name]
    if per_adapter:
        filename += "_per_adapter"
        keep += ["adapter"]
    to_save = df[keep]

    if per_adapter:
        to_save = to_save.pivot(
            index=["validator", "validator_args", "task"], columns="adapter"
        )
        to_save = to_save.droplevel(0, axis=1).rename_axis(None, axis=1).reset_index()

    save_df(output_folder, df, to_save, filename)


def get_weighted_spearman_score(output_folder, df, per_adapter, src_threshold):
    def score_fn(x):
        return weighted_spearman(x[TARGET_ACCURACY].values, x["score"].values, pow=2)

    get_correlation(
        output_folder, df, per_adapter, src_threshold, score_fn, "weighted_spearman"
    )


def get_spearman_score(output_folder, df, per_adapter, src_threshold):
    def score_fn(x):
        return spearman(x[TARGET_ACCURACY].values, x["score"].values)

    get_correlation(output_folder, df, per_adapter, src_threshold, score_fn, "spearman")


def get_best_accuracy_per_adapter(output_folder, df, nlargest):
    # Filtering by validator not actually necessary. I'm only doing it to remove duplicates
    to_save = df[
        (df["validator"] == "Accuracy")
        & (df["validator_args"] == '{"average": "micro", "split": "target_train"}')
    ]
    groupby = group_by_task(per_adapter=True)
    to_save = (
        to_save.groupby(groupby + ["trial_num"])[TARGET_ACCURACY].max().reset_index()
    )
    ranked = to_save.groupby(groupby)[TARGET_ACCURACY].rank(
        method="min", ascending=False
    )
    to_save = to_save[ranked <= nlargest]
    to_save = to_save.groupby(groupby, as_index=False).agg(
        {TARGET_ACCURACY: ["mean", "std"]}
    )
    to_save.columns = to_save.columns.map("".join)
    to_save = to_save.rename(
        columns={
            f"{TARGET_ACCURACY}mean": TARGET_ACCURACY,
            f"{TARGET_ACCURACY}std": f"{TARGET_ACCURACY}_std",
        }
    )
    to_save = add_task_column(to_save)
    to_save = to_save[["adapter", "task", TARGET_ACCURACY, f"{TARGET_ACCURACY}_std"]]
    save_df(output_folder, df, to_save, f"best_accuracy_per_adapter_{nlargest}")


def get_fn(args):
    def eval_validators(output_folder, df):
        for s in args.src_threshold:
            get_weighted_spearman_score(output_folder, df, False, s)
            get_weighted_spearman_score(output_folder, df, True, s)
            get_spearman_score(output_folder, df, False, s)
            get_spearman_score(output_folder, df, True, s)
        get_best_accuracy_per_adapter(output_folder, df, args.nlargest)

    return eval_validators


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--output_folder", type=str, default="tables")
    parser.add_argument("--nlargest", type=int, default=5)
    parser.add_argument("--src_threshold", nargs="+", type=float, default=[0.0])
    create_main.add_main_args(parser)
    args = parser.parse_args()
    create_main.main(args, get_fn(args), get_fn(args))
