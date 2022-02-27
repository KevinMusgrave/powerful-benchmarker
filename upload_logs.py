import argparse
import os
import subprocess
import sys

sys.path.insert(0, "src")
from powerful_benchmarker.utils.constants import add_default_args


def main(cfg):
    curr_dir = os.path.abspath(os.path.dirname(__file__))

    command = "bash -i ./misc/upload_logs.sh {0} {1} {2} {3} {4} {5}".format(
        cfg.exp_folder,
        cfg.slurm_folder,
        cfg.gdrive_folder,
        cfg.sleep_time,
        curr_dir,
        cfg.conda_env,
    )
    subprocess.run(command.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(
        parser, ["exp_folder", "slurm_folder", "conda_env", "gdrive_folder"]
    )
    parser.add_argument("--sleep_time", type=str, default="240m")
    args = parser.parse_args()
    main(args)
