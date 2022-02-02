import argparse
import os
import subprocess

import yaml


def main(cfg):
    all_jobids_filename = os.path.join(cfg.root_experiment_folder, "all_jobids.txt")
    with open(all_jobids_filename, "r") as f:
        jobids = " ".join([line.rstrip("\n") for line in f])

    command = f"scancel {jobids}"
    print("killing slurm jobs")
    subprocess.run(command.split(" "))
    print("deleting all_jobids.txt")
    os.remove(all_jobids_filename)


if __name__ == "__main__":
    with open("constants.yaml", "r") as f:
        constants = yaml.safe_load(f)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--root_experiment_folder",
        type=str,
        default=constants["experiment_folder"],
    )
    args = parser.parse_args()
    main(args)
