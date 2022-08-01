import argparse
import os
import sys

sys.path.insert(0, ".")
from pytorch_adapt.utils import common_functions as c_f

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import create_main
from validator_tests.utils.constants import TARGET_ACCURACY, add_exp_group_args
from validator_tests.utils.df_utils import get_name_from_df, get_sorted_unique
from validator_tests.utils.trndcg import weighted_spearman


def save_df(folder, df, per_adapter):
    folder = os.path.join(folder, get_name_from_df(df, assert_one_task=True))
    c_f.makedir_if_not_there(folder)
    filename = "trndcg"
    keep = ["validator", "validator_args", "task", "trndcg"]
    if per_adapter:
        filename += "_per_adapter"
        keep += ["adapter"]
    df = df[keep]

    if per_adapter:
        df = df.pivot(index=["validator", "validator_args", "task"], columns="adapter")
        df = df.droplevel(0, axis=1).rename_axis(None, axis=1).reset_index()

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


def get_trndcg_score(output_folder, df, per_adapter):
    new_df = df.groupby(group_by_task_validator(per_adapter))[
        [TARGET_ACCURACY, "score"]
    ].apply(
        lambda x: weighted_spearman(x[TARGET_ACCURACY].values, x["score"].values, pow=2)
    )
    new_df = new_df.reset_index(name="trndcg")
    df = assign_original_df_info(new_df, df)
    save_df(output_folder, df, per_adapter)


def eval_validators(output_folder, df):
    get_trndcg_score(output_folder, df, False)
    get_trndcg_score(output_folder, df, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--output_folder", type=str, default="tables")
    create_main.add_main_args(parser)
    args = parser.parse_args()
    create_main.main(args, eval_validators, eval_validators)
