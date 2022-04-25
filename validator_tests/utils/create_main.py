import os
import sys

sys.path.insert(0, ".")
from validator_tests.utils import utils
from validator_tests.utils.df_utils import (
    get_exp_groups_with_matching_tasks,
    get_processed_df,
)


def main(args, fn1, fn2):
    exp_groups = utils.get_exp_groups(args)
    output_folder = getattr(args, "output_folder", None)
    for e in exp_groups:
        print("exp_group", e)
        # do per feature layer
        exp_folder = os.path.join(args.exp_folder, e)
        df = get_processed_df(exp_folder)
        if df is not None:
            fn1(exp_folder, [e], output_folder, df)

    # now do with feature layers in one dataframe
    # which are saved in args.exp_folder
    combined_dfs, combined_exp_groups = get_exp_groups_with_matching_tasks(
        args.exp_folder, exp_groups
    )
    for df, e in zip(combined_dfs, combined_exp_groups):
        print("exp_groups", e)
        fn2(args.exp_folder, e, output_folder, df)


def add_topN_args(parser):
    parser.add_argument("--topN", type=int, default=200)
    parser.add_argument("--topN_per_adapter", type=int, default=20)
