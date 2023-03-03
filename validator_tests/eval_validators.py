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


def get_score_fn(name):
    def ws_score_fn(x):
        return weighted_spearman(x["score"].values, x[TARGET_ACCURACY].values, pow=2)

    def s_score_fn(x):
        return spearman(x["score"].values, x[TARGET_ACCURACY].values)

    return {"weighted_spearman": ws_score_fn, "spearman": s_score_fn}[name]


def _get_correlation(df, per_adapter, src_threshold, name, score_fn=None):
    score_fn = score_fn if score_fn else get_score_fn(name)

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

    keep = ["validator", "validator_args", "task", name]
    if per_adapter:
        keep += ["adapter"]
    to_save = df[keep]

    if per_adapter:
        to_save = to_save.pivot(
            index=["validator", "validator_args", "task"], columns="adapter"
        )
        to_save = to_save.droplevel(0, axis=1).rename_axis(None, axis=1).reset_index()

    return to_save


def get_correlation(output_folder, df, per_adapter, src_threshold, name):
    to_save = _get_correlation(df, per_adapter, src_threshold, name)
    filename = f"{name}_{src_threshold}_src_threshold"
    if per_adapter:
        filename += "_per_adapter"
    save_df(output_folder, df, to_save, filename)


def _get_best_accuracy_per_adapter(
    df, nlargest, rank_by=TARGET_ACCURACY, return_ranks=False
):
    assert rank_by in [TARGET_ACCURACY, "score"]
    groupby = group_by_task_validator(per_adapter=True)
    groupby_with_fl_trial_num = groupby + ["feature_layer", "trial_num"]

    # best score per feature layer per trial
    ranked = df.groupby(groupby_with_fl_trial_num)[rank_by].rank(
        method="min", ascending=False
    )
    to_save = df[ranked <= 1]

    # remove duplicate scores for a trial by taking the earliest epoch
    to_save = to_save.sort_values(by=["epoch"]).drop_duplicates(
        subset=groupby_with_fl_trial_num
    )
    # for oracle (TARGET_ACCURACY), we want only the first tied checkpoint
    # for validation scores, we want tied checkpoints to share the same rank
    rank_method = "first" if rank_by == TARGET_ACCURACY else "min"
    # best scores across trials
    ranked = to_save.groupby(groupby)[rank_by].rank(method=rank_method, ascending=False)
    if return_ranks:
        to_save["rank"] = ranked
        return to_save[to_save["rank"] <= nlargest]

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
    keep_cols = [
        "adapter",
        "task",
        "validator",
        "validator_args",
        TARGET_ACCURACY,
        f"{TARGET_ACCURACY}_std",
    ]
    if rank_by == TARGET_ACCURACY:
        to_save = to_save[
            (to_save["validator"] == "Accuracy")
            & (
                to_save["validator_args"]
                == '{"average": "micro", "split": "target_train"}'
            )
        ]
        keep_cols.remove("validator")
        keep_cols.remove("validator_args")

    return to_save[keep_cols]


def get_best_accuracy_per_adapter(output_folder, df, nlargest, rank_by=TARGET_ACCURACY):
    to_save = _get_best_accuracy_per_adapter(df, nlargest, rank_by)
    rank_by_str = "" if rank_by == TARGET_ACCURACY else f"_ranked_by_{rank_by}"
    save_df(
        output_folder, df, to_save, f"best_accuracy_per_adapter{rank_by_str}_{nlargest}"
    )


def eval_validators(output_folder, df, src_thresholds, nlargest):
    for s in src_thresholds:
        get_correlation(output_folder, df, False, s, "weighted_spearman")
        get_correlation(output_folder, df, True, s, "weighted_spearman")
        get_correlation(output_folder, df, False, s, "spearman")
        get_correlation(output_folder, df, True, s, "spearman")
    get_best_accuracy_per_adapter(output_folder, df, nlargest)
    get_best_accuracy_per_adapter(output_folder, df, nlargest, rank_by="score")


def get_fn(args):
    def fn(output_folder, df):
        eval_validators(output_folder, df, args.src_threshold, args.nlargest)

    return fn


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
