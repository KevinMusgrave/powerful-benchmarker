import argparse
import os
import subprocess
import sys

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args


def main(cfg):
    curr_dir = os.getcwd()

    command = "bash -i ./scripts/upload_dfs.sh {0} {1} {2}".format(
        cfg.exp_folder, cfg.gdrive_folder, curr_dir
    )
    subprocess.run(command.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "gdrive_folder"])
    args = parser.parse_args()
    main(args)
