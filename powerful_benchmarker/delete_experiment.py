import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils
from validator_tests.utils.constants import add_exp_group_args


def delete_experiment(cfg, exp_group):
    experiment_paths = sorted(glob.glob(f"{cfg.exp_folder}/{exp_group}/*"))
    num_folders = 0
    for p in experiment_paths:
        if not os.path.isdir(p):
            continue
        if os.path.basename(p) == cfg.adapter:
            num_folders += 1
            if cfg.delete:
                print("deleting", p)
                shutil.rmtree(p)
    print("num_folders", num_folders)


def main(cfg):
    exp_groups = utils.get_exp_groups(cfg)
    for e in exp_groups:
        delete_experiment(cfg, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--adapter", type=str, default="")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    main(args)
