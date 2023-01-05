import os

import numpy as np
import pandas as pd

from latex.correlation import base_filename
from latex.correlation import get_preprocess_df as get_preprocess_df_correlation
from latex.correlation_single_adapter import (
    get_postprocess_df as get_postprocess_df_correlation,
)
from latex.table_creator import table_creator
from validator_tests.utils import utils


def get_best_validators(args, name, src_threshold):
    basename = base_filename(name, True, src_threshold)
    exp_groups = utils.get_exp_groups(args, args.input_folder)

    dfs, output_folder = table_creator(
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
        best_validators[adapter] = {}
        for task in df.columns:
            if task in ["Mean", "Std"]:
                continue
            best_validators[adapter][task] = df.loc[df[task].idxmax()].name

    return best_validators, output_folder


def best_validator_per_adapter_task(args, name, src_threshold):
    best_validators, output_folder = get_best_validators(args, name, src_threshold)

    df = pd.DataFrame.from_dict(best_validators).transpose()

    for c in df.columns:
        df[c] = df[c].agg(" ".join)

    split_columns = np.array_split(df.columns, 5)
    for i, c in enumerate(split_columns):
        df[c].style.to_latex(
            os.path.join(output_folder, f"best_validator_per_adapter_task_{i}.tex"),
            hrules=True,
            position_float="centering",
        )
