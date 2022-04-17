import argparse
import os
import subprocess

import submitit

from powerful_benchmarker.utils.constants import add_default_args
from powerful_benchmarker.utils.utils import create_slurm_args


def exp_launcher(args, command):
    full_command = f"bash -i ./scripts/script_wrapper.sh {args.conda_env}".split(" ")
    full_command += [command]
    subprocess.run(full_command)


def main(args, slurm_args):
    executor = submitit.AutoExecutor(
        folder=os.path.join(args.exp_folder, args.slurm_folder)
    )
    executor.update_parameters(
        timeout_min=0,
        tasks_per_node=1,
        slurm_additional_parameters=slurm_args,
    )
    job = executor.submit(exp_launcher, args, args.command)
    print("started", job.job_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "conda_env", "slurm_folder"])
    parser.add_argument("--command", type=str, required=True)
    parser.add_argument("--slurm_config_folder", type=str, required=True)
    parser.add_argument("--slurm_config", type=str, required=True)
    args, unknown_args = parser.parse_known_args()
    slurm_args = create_slurm_args(args, unknown_args, args.slurm_config_folder)
    main(args, slurm_args)
