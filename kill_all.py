import argparse
import os
import subprocess
import sys

sys.path.insert(0, "src")
from powerful_benchmarker.utils.constants import add_default_args


def main(cfg):
    all_jobids_filename = os.path.join(cfg.exp_folder, "all_jobids.txt")
    with open(all_jobids_filename, "r") as f:
        jobids = " ".join([line.rstrip("\n") for line in f])

    command = f"scancel {jobids}"
    print("killing slurm jobs")
    subprocess.run(command.split(" "))
    print("deleting all_jobids.txt")
    os.remove(all_jobids_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    args = parser.parse_args()
    main(args)
