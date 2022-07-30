import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, ".")

from validator_tests.utils import utils
from validator_tests.utils.constants import add_exp_group_args


def agg(df):
    return df.groupby(["validator", "validator_args", "task"]).agg(["mean", "std"])


def main(args):
    exp_groups = utils.get_exp_groups(args, exp_folder=args.input_folder)

    for e in exp_groups:
        folder = os.path.join(args.input_folder, e)
        for per_adapter in [False, True]:
            for threshold in [0, 0.5]:
                curr_df = []
                topN_list = args.topN_per_adapter if per_adapter else args.topN
                per_adapter_str = "per_adapter_" if per_adapter else ""
                for topN in topN_list:
                    curr_files = glob.glob(
                        os.path.join(
                            folder,
                            f"predicted_best_acc_top{topN}_{per_adapter_str}{threshold}_*.pkl",
                        )
                    )
                    assert len(curr_files) == 1
                    curr_df.append(pd.read_pickle(curr_files[0]))
                if len(curr_df) == 0:
                    continue
                curr_df = pd.concat(curr_df, axis=0)

                if per_adapter:
                    curr_df = curr_df.drop(
                        ["predicted_best_acc_std"], axis=1, level=0
                    ).droplevel(axis=1, level=0)
                    curr_df = agg(curr_df).swaplevel(axis=1).sort_index(axis=1, level=0)
                else:
                    curr_df = curr_df.drop(columns=["predicted_best_acc_std"])
                    curr_df = agg(curr_df).droplevel(axis=1, level=0)
                    curr_df = curr_df.rename(
                        columns={
                            "mean": "predicted_best_acc_mean",
                            "std": "predicted_best_acc_std",
                        }
                    )

                curr_df = curr_df.reset_index()

                filename = f"averaged_predicted_best_acc_top{'_'.join(topN_list)}_{per_adapter_str}{threshold}_src_threshold"
                filename = os.path.join(folder, filename)
                curr_df.to_csv(f"{filename}.csv", index=False)
                curr_df.to_pickle(f"{filename}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_exp_group_args(parser)
    parser.add_argument("--input_folder", type=str, default="tables")
    parser.add_argument("--topN", type=str, nargs="+", default=[])
    parser.add_argument("--topN_per_adapter", type=str, nargs="+", default=[])
    args = parser.parse_args()
    main(args)
