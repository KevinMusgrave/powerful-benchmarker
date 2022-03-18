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
    utils.count_pkls(os.path.join(cfg.exp_folder, cfg.exp_group), cfg.validator, fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--exp_group", type=str, required=True)
    parser.add_argument("--validator", type=str, default="")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    main(args)
