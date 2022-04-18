import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils
from validator_tests.utils.constants import (
    ALL_DFS_FILENAME,
    VALIDATOR_TESTS_FOLDER,
    add_exp_group_args,
)


def collect_dfs(args, exp_group):
    df = []
    exp_folder = os.path.join(args.exp_folder, exp_group)
    exp_names = [os.path.basename(x) for x in glob.glob(os.path.join(exp_folder, "*"))]
    if args.slurm_folder in exp_names:
        exp_names.remove(args.slurm_folder)
    for e in exp_names:
        exp_folders = utils.get_exp_folders(exp_folder, e)
        for ef in exp_folders:
            print(ef, flush=True)
            df_files = glob.glob(os.path.join(ef, VALIDATOR_TESTS_FOLDER, "*.pkl"))
            for dff in df_files:
                df.append(pd.read_pickle(dff))

    df = pd.concat(df, axis=0, ignore_index=True)
    filename = os.path.join(exp_folder, ALL_DFS_FILENAME)
    df.to_pickle(filename)


def main(args):
    exp_groups = utils.get_exp_groups(args)
    for e in exp_groups:
        collect_dfs(args, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "slurm_folder"])
    add_exp_group_args(parser)
    args = parser.parse_args()
    main(args)
