import os

import pandas as pd

from latex import utils as latex_utils
from latex.best_accuracy_per_adapter import (
    best_accuracy_per_adapter,
    min_value_fn,
    reshape_into_best_accuracy_table,
)
from latex.correlation import base_filename, filter_and_process_validator_args
from latex.correlation import get_preprocess_df as get_preprocess_df_correlation
from latex.correlation_single_adapter import (
    get_postprocess_df as get_postprocess_df_correlation,
)
from latex.table_creator import table_creator
from validator_tests.utils import utils
from validator_tests.utils.constants import TARGET_ACCURACY


def get_postprocess_df(best_validators):
    def fn(df):
        df = pd.concat(df, axis=0)
        latex_utils.convert_adapter_name(df)
        df = latex_utils.rename_validator_args(df)
        filter = False
        for k, v in best_validators.items():
            filter |= (
                (df["adapter"] == k)
                & (df["validator"] == v[0])
                & (df["validator_args"] == v[1])
            )
        df = df[filter]
        df = df.drop(columns=[f"{TARGET_ACCURACY}_std", "validator", "validator_args"])
        return reshape_into_best_accuracy_table(df)

    return fn


def get_best_validators(args, name, src_threshold):
    basename = base_filename(name, True, src_threshold)
    exp_groups = utils.get_exp_groups(args, args.input_folder, "_select_best")

    dfs, _ = table_creator(
        args,
        args.input_folder,
        args.output_folder,
        basename,
        preprocess_df=get_preprocess_df_correlation(per_adapter=True),
        postprocess_df=get_postprocess_df_correlation(remove_index_names=False),
        do_save_to_latex=False,
        exp_groups=exp_groups,
    )

    best_validators = {}
    for adapter, df in dfs.items():
        df = df.loc[df["Mean"].idxmax()]
        best_validators[adapter] = df.name + (rf"${str(df.Mean)} \pm {str(df.Std)}$",)

    tasks = [x for x in list(dfs.values())[0].columns if x not in ["Mean", "Std"]]

    return best_validators, tasks


def get_meanstd(df, name):
    mean = df.mean(axis=1).round(1).astype(str)
    std = df.std(axis=1).round(1).astype(str)
    return (
        ("$" + mean + r" \pm " + std + "$")
        .to_frame(name)
        .reset_index()
        .rename(columns={"index": "Algorithm"})
    )


def pred_acc_using_best_adapter_validator_pairs(args, name, src_threshold):
    best_validators, tasks = get_best_validators(args, name, src_threshold)

    nlargest = args.nlargest
    basename = f"best_accuracy_per_adapter_ranked_by_score_{nlargest}"
    color_map_tag_kwargs = {
        "tag_prefix": latex_utils.get_tag_prefix(basename),
        "min_value_fn": min_value_fn,
    }
    best_accs, output_folder = table_creator(
        args,
        args.input_folder,
        args.output_folder,
        basename,
        preprocess_df=filter_and_process_validator_args,
        postprocess_df=get_postprocess_df(best_validators),
        color_map_tag_kwargs=color_map_tag_kwargs,
        add_resizebox=True,
        final_str_hook=latex_utils.adapter_final_str_hook,
    )

    best_accs_oracle, _ = best_accuracy_per_adapter(args, do_save_to_latex=False)

    tasks = [x for x in tasks if x in best_accs.columns]
    best_accs = best_accs[tasks]
    best_accs_oracle = best_accs_oracle[tasks]
    diff_from_oracle = best_accs - best_accs_oracle

    best_accs_meanstd = get_meanstd(best_accs, "Average Accuracy")
    diff_from_oracle_meanstd = get_meanstd(diff_from_oracle, "Degradation")

    to_save = (
        pd.DataFrame(best_validators)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "Algorithm",
                0: "Validator",
                1: "Validator Parameters",
                2: "Weighted Spearman Correlation",
            }
        )
    )

    to_save = (
        to_save.merge(best_accs_meanstd)
        .drop(columns=["Weighted Spearman Correlation"])
        .sort_values(by="Average Accuracy", ascending=False)
    )

    to_save = to_save.merge(diff_from_oracle_meanstd)

    to_save.style.hide(axis="index").to_latex(
        os.path.join(output_folder, "best_validator_per_algorithm.tex"),
        hrules=True,
        position_float="centering",
    )
