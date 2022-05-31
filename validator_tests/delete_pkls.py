import argparse
import os
import sys

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils
from validator_tests.utils.constants import add_exp_group_args


def delete_fn(pkls):
    for pkl in pkls:
        print("deleting", pkl)
        os.remove(pkl)


def main(cfg):
    fn = delete_fn if cfg.delete else None
    exp_groups = utils.get_exp_groups(cfg)

    for exp_group in exp_groups:
        utils.count_pkls(os.path.join(cfg.exp_folder, exp_group), cfg.validator, fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--validator", type=str, default="")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    main(args)
