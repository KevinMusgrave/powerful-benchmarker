import argparse
import os
import sys
from functools import partialmethod

import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import create_main
from validator_tests.utils.constants import add_exp_group_args
from validator_tests.utils.df_utils import get_per_src_basename
from validator_tests.utils.threshold_utils import (
    convert_predicted_best_acc_to_rel,
    get_all_per_task_validator,
    get_all_per_task_validator_adapter,
    get_per_threshold,
)


def create_per_x_threshold(df, exp_folder, per_adapter, topN, exp_groups):
    if topN is None:
        return
    print(f"per_adapter = {per_adapter}")
    fn = (
        get_all_per_task_validator_adapter(topN)
        if per_adapter
        else get_all_per_task_validator(topN)
    )
    per_src = get_per_threshold(df, fn)
    per_src = convert_predicted_best_acc_to_rel(
        df, per_src, per_adapter, topN, len(exp_groups)
    )
    basename = get_per_src_basename(per_adapter, topN, df=per_src)
    filename = os.path.join(exp_folder, basename)
    print(f"saving to {filename}\n\n")
    per_src.to_pickle(filename)


def run_both(exp_folder, exp_groups, output_folder, df, topN, topN_per_adapter):
    create_per_x_threshold(df, exp_folder, False, topN, exp_groups)
    create_per_x_threshold(df, exp_folder, True, topN_per_adapter, exp_groups)


def get_fn(args):
    def fn(*in_args):
        run_both(*in_args, args.topN, args.topN_per_adapter)

    return fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    create_main.add_main_args(parser)
    args = parser.parse_args()
    create_main.main(args, get_fn(args), get_fn(args))
