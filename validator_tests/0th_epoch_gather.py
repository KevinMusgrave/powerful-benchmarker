import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from pytorch_adapt.models.pretrained_scores import (
    pretrained_src_accuracy,
    pretrained_target_accuracy,
)

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import df_utils, utils
from validator_tests.utils.constants import (
    ALL_DFS_FILENAME,
    VALIDATOR_TESTS_FOLDER,
    add_exp_group_args,
)


def collect_dfs(args, exp_group):
    df = []
    exp_folder = os.path.join(args.exp_folder, exp_group)
    exp_folders = utils.get_exp_folders(exp_folder, "epoch_0", use_glob=True)
    for ef in exp_folders:
        print(ef, flush=True)
        for v in args.validators:
            search_term = f"{v}*.pkl"
            df_files = glob.glob(os.path.join(ef, VALIDATOR_TESTS_FOLDER, search_term))
            for dff in df_files:
                curr_df = pd.read_pickle(dff)
                curr_df = curr_df[curr_df["epoch"] == "0"]
                df.append(curr_df)

    if len(df) > 0:
        df = pd.concat(df, axis=0, ignore_index=True)

        # check that the computed accuracies match the ones provided by pytorch adapt
        for split in df_utils.SPLIT_NAMES:
            for average in ["micro", "macro"]:
                acc = df_utils.get_acc_df(df, split, average)
                dataset = acc["dataset"].item()
                src_domains = acc["src_domains"].item()
                target_domains = acc["target_domains"].item()
                if split.startswith("src"):
                    correct = pretrained_src_accuracy(
                        dataset, src_domains, split.replace("src_", ""), average
                    )
                else:
                    correct = pretrained_target_accuracy(
                        dataset,
                        src_domains,
                        target_domains,
                        split.replace("target_", ""),
                        average,
                    )

                assert (
                    np.round(
                        acc[df_utils.acc_score_column_name(split, average)].item(), 4
                    )
                    == correct
                )

        all_dfs = pd.read_pickle(os.path.join(exp_folder, ALL_DFS_FILENAME))
        df = pd.concat([all_dfs, df], axis=0)
        df.to_pickle(os.path.join(exp_folder, ALL_DFS_FILENAME))


def main(args):
    exp_groups = utils.get_exp_groups(args)
    for e in exp_groups:
        collect_dfs(args, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "slurm_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--validators", nargs="+", type=str, default=[""])
    args = parser.parse_args()
    main(args)
