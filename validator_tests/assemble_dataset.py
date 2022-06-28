import glob
import os
import sys

sys.path.insert(0, ".")

import argparse

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils
from validator_tests.utils.constants import add_exp_group_args


def assemble_fn(epoch, x, exp_config, exp_folder):
    # TODO: save data to a new hdf5 file
    pass


# Full dataset is too large.
# Keep every 20th trial.
def condition(iteration, *_):
    if iteration % 20 == 0:
        return True
    return False


def assemble(args, exp_group):
    exp_folder = os.path.join(args.exp_folder, exp_group)
    exp_names = [
        os.path.basename(x)
        for x in glob.glob(os.path.join(exp_folder, "*"))
        if os.path.isdir(x)
    ]
    if args.slurm_folder in exp_names:
        exp_names.remove(args.slurm_folder)
    for e in exp_names:
        exp_folders = utils.get_exp_folders(exp_folder, e)
        utils.apply_to_data(exp_folders, condition, fn=assemble_fn)


def main(args):
    exp_groups = utils.get_exp_groups(args)
    for e in exp_groups:
        assemble(args, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "slurm_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--output_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
