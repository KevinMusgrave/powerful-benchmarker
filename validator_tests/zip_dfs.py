import argparse
import subprocess

from powerful_benchmarker.utils.constants import add_default_args


def main(cfg):
    command = "bash -i ./validator_tests/scripts/zip_dfs.sh {0}".format(cfg.exp_folder)
    subprocess.run(command.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    args = parser.parse_args()
    main(args)
