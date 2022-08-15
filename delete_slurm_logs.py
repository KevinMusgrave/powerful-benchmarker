import argparse
import glob
import os

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils
from validator_tests.utils.constants import add_exp_group_args


def main(cfg):
    if not cfg.slurm_folder:
        raise ValueError
    exp_groups = utils.get_exp_groups(cfg)
    num_files = 0
    for e in exp_groups:
        slurm_folder = os.path.join(cfg.exp_folder, e, cfg.slurm_folder)
        if not os.path.isdir(slurm_folder):
            continue
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
    add_exp_group_args(parser)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    main(args)
