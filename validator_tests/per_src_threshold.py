import argparse
import os
import sys

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils.constants import get_per_src_basename, get_processed_df
from validator_tests.utils.threshold_utils import (
    convert_predicted_best_acc_to_rel,
    get_all_per_task_validator,
    get_all_per_task_validator_adapter,
    get_per_threshold,
)


def create_per_x_threshold(df, exp_folder, per_adapter, nlargest):
    print(f"per_adapter = {per_adapter}")
    basename = get_per_src_basename(per_adapter)
    filename = os.path.join(exp_folder, basename)
    fn = (
        get_all_per_task_validator_adapter(nlargest)
        if per_adapter
        else get_all_per_task_validator(nlargest)
    )
    per_src = get_per_threshold(df, fn)
    per_src = convert_predicted_best_acc_to_rel(df, per_src, per_adapter, nlargest)
    per_src.to_pickle(filename)


def main(args):
    exp_folder = os.path.join(args.exp_folder, args.exp_group)
    df = get_processed_df(exp_folder)
    create_per_x_threshold(df, exp_folder, False, 100)
    create_per_x_threshold(df, exp_folder, True, 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--exp_group", type=str, required=True)
    args = parser.parse_args()
    main(args)
