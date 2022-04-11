import argparse
import glob
import os
import sys

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils


def main(cfg):
    if not cfg.slurm_folder:
        raise ValueError
    exp_groups = cfg.exp_groups
    if not exp_groups:
        exp_groups = utils.filter_exp_groups(
            args.exp_folder,
            prefix=args.exp_group_prefix,
            suffix=args.exp_group_suffix,
            contains=args.exp_group_contains,
        )
    num_files = 0
    for e in exp_groups:
        slurm_folder = os.path.join(cfg.exp_folder, e, cfg.slurm_folder)
        if not os.path.isdir(slurm_folder):
            raise ValueError
        files = glob.glob(os.path.join(slurm_folder, "*"))
        num_files += len(files)
        if cfg.delete:
            for f in files:
                print("deleting", f)
                os.remove(f)
    print("num files =", num_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "slurm_folder"])
    parser.add_argument("--exp_groups", nargs="+", type=str, default=[])
    parser.add_argument("--exp_group_prefix", type=str)
    parser.add_argument("--exp_group_suffix", type=str)
    parser.add_argument("--exp_group_contains", type=str)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    main(args)
