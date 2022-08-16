import argparse
import os
import subprocess
import sys

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import JOBIDS_FILENAME, add_default_args
from validator_tests.utils.constants import JOBIDS_FILENAME as V_JOBSID_FILENAME


def main(cfg):
    curr_dir = os.getcwd()

    command = "bash -i ./scripts/upload_logs.sh {0} {1} {2} {3} {4}".format(
        cfg.exp_folder,
        cfg.gdrive_folder,
        cfg.sleep_time,
        curr_dir,
        cfg.conda_env,
    )
    command = command.split(" ")
    command.append(f"{JOBIDS_FILENAME} {V_JOBSID_FILENAME}")
    subprocess.run(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "conda_env", "gdrive_folder"])
    parser.add_argument("--sleep_time", type=str, default="120m")
    args = parser.parse_args()
    main(args)
