import argparse
import os
import sys

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils


def delete_fn(pkls):
    for pkl in pkls:
        print("deleting", pkl)
        os.remove(pkl)


def main(cfg):
    fn = delete_fn if cfg.delete else None
    exp_groups = cfg.exp_groups
    if not exp_groups:
        exp_groups = utils.filter_exp_groups(
            args.exp_folder,
            prefix=args.exp_group_prefix,
            suffix=args.exp_group_suffix,
            contains=args.exp_group_contains,
        )

    for exp_group in exp_groups:
        utils.count_pkls(os.path.join(cfg.exp_folder, exp_group), cfg.validator, fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--exp_groups", nargs="+", type=str, default=[])
    parser.add_argument("--exp_group_prefix", type=str)
    parser.add_argument("--exp_group_suffix", type=str)
    parser.add_argument("--exp_group_contains", type=str)
    parser.add_argument("--validator", type=str, default="")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    main(args)
