import argparse
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "validator_tests/src")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--exp_group", type=str, required=True)
    parser.add_argument("--validator", type=str, default="")
    args = parser.parse_args()
    utils.count_pkls(os.path.join(args.exp_folder, args.exp_group), args.validator)
