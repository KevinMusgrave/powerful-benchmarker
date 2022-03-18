import argparse
import os
import subprocess

from .utils.constants import add_default_args


def main(cfg):
    curr_dir = os.getcwd()

    command = "bash -i ./powerful_benchmarker/scripts/upload_logs.sh {0} {1} {2} {3} {4} {5}".format(
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
    parser.add_argument("--sleep_time", type=str, default="120m")
    args = parser.parse_args()
    main(args)
