import argparse
import os
import sys

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.eval_validators import eval_validators
from validator_tests.utils import create_main
from validator_tests.utils.constants import add_exp_group_args


def eval_subsets(output_folder, df):
    epoch_intervals = list(range(1, 11))
    run_intervals = list(range(1, 11))
    for e in epoch_intervals:
        for r in run_intervals:
            print(e, r)
            curr_df = df[
                (df["epoch"].astype(int) % e == 0)
                & (df["trial_num"].astype(int) % r == 0)
            ]
            curr_folder = os.path.join(output_folder, f"e{e}_r{r}")
            eval_validators(curr_folder, curr_df, [0], 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--output_folder", type=str, default="tables_subsets")
    create_main.add_main_args(parser)
    args = parser.parse_args()
    create_main.main(args, eval_subsets, eval_subsets)
