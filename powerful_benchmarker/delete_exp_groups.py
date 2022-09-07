import argparse
import os
import shutil
import sys
from pprint import pprint

sys.path.insert(0, ".")

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import utils
from validator_tests.utils.constants import add_exp_group_args


def main(cfg):
    exp_groups = utils.get_exp_groups(cfg)
    folders_to_delete = []
    for e in exp_groups:
        full_path = os.path.join(cfg.exp_folder, e)
        if cfg.delete:
            print("deleting", full_path)
            shutil.rmtree(full_path)
        else:
            folders_to_delete.append(full_path)

    if not cfg.delete:
        print("Preview of folders to delete")
        pprint(folders_to_delete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    main(args)
