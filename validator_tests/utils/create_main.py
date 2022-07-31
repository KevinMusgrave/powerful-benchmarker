import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from validator_tests.utils import utils
from validator_tests.utils.df_utils import (
    get_exp_groups_with_matching_tasks,
    get_processed_df,
)


def main(args, fn1, fn2):
    exp_groups = utils.get_exp_groups(args)
    output_folder = getattr(args, "output_folder", None)
    if args.run_single:
        for e in exp_groups:
            print("exp_group", e)
            # do per feature layer
            exp_folder = os.path.join(args.exp_folder, e)
            df = get_processed_df(exp_folder)
            if df is not None:
                fn1(exp_folder, [e], output_folder, df)

    if args.run_combined:
        # now do with feature layers in one dataframe
        # which are saved in args.exp_folder
        print("finding combined dfs")
        combined_exp_groups = get_exp_groups_with_matching_tasks(
            args.exp_folder, exp_groups
        )
        for e_group in combined_exp_groups:
            print("exp_groups", e_group)
            df = []
            for e in e_group:
                df.append(get_processed_df(os.path.join(args.exp_folder, e)))
            df = pd.concat(df, axis=0, ignore_index=True)
            fn2(args.exp_folder, e_group, output_folder, df)


def add_main_args(parser):
    parser.add_argument("--run_single", action="store_true")
    parser.add_argument("--run_combined", action="store_true")
