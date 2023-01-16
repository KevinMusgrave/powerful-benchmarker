# number of checkpoints can be calculated manually
# but some checkpoints may have been corrupted somehow etc.
# So here we calculate the actual number of checkpoints used when evaluating the validators
import argparse
import glob
import sys

import pandas as pd

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils.constants import PROCESSED_DF_FILENAME


def main(exp_folder):
    processed = glob.glob(f"{exp_folder}/**/{PROCESSED_DF_FILENAME}")
    num_checkpoints = 0
    for p in processed:
        df = pd.read_pickle(p)
        df = df[["epoch", "trial_num", "adapter"]].drop_duplicates()
        num_checkpoints += len(df)
        print(p, len(df))
    print(f"num_checkpoints = {num_checkpoints}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    args = parser.parse_args()
    main(args.exp_folder)
