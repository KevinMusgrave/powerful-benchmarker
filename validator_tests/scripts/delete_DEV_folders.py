import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils.constants import VALIDATOR_TESTS_FOLDER, add_exp_group_args
from validator_tests.utils.utils import get_exp_groups


def main(args):
    exp_groups = get_exp_groups(args)
    num_folders = 0
    for e in exp_groups:
        for exp_name in args.exp_names:
            folder = os.path.join(args.exp_folder, e, exp_name)
            dev_folders = glob.glob(
                os.path.join(folder, "*", VALIDATOR_TESTS_FOLDER, "DEV*")
            )
            for dev_folder in dev_folders:
                if os.path.isdir(dev_folder) and not dev_folder.endswith(".pkl"):
                    num_folders += 1
                    if args.delete:
                        print("deleting", dev_folder)
                        shutil.rmtree(dev_folder)
    print("num folders =", num_folders)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--exp_names", nargs="+", type=str, required=True)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    main(args)
