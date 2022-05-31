import argparse
import os
import subprocess

import submitit

from powerful_benchmarker.utils.constants import add_default_args
from powerful_benchmarker.utils.utils import create_slurm_args
from validator_tests.utils import utils
from validator_tests.utils.constants import add_exp_group_args, exp_group_args


def exp_launcher(conda_env, command):
    full_command = f"bash -i ./scripts/script_wrapper.sh {conda_env}".split(" ")
    full_command += [command]
    subprocess.run(full_command)


def run(args, slurm_args, exp_group):
    executor = submitit.AutoExecutor(
        folder=os.path.join(args.exp_folder, args.slurm_folder)
    )
    executor.update_parameters(
        timeout_min=0,
        tasks_per_node=1,
        slurm_additional_parameters=slurm_args,
    )

    command = args.command
    if exp_group:
        command = f"{command} --exp_groups {exp_group}"

    job = executor.submit(exp_launcher, args.conda_env, command)
    print("started", job.job_id)


def main(args, slurm_args):
    if not any(getattr(args, k) for k in exp_group_args()):
        run(args, slurm_args, None)
    else:
        for e in utils.get_exp_groups(args):
            run(args, slurm_args, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "conda_env", "slurm_folder"])
    add_exp_group_args(parser)
    parser.add_argument("--command", type=str, required=True)
    parser.add_argument("--slurm_config_folder", type=str, required=True)
    parser.add_argument("--slurm_config", type=str, required=True)
    args, unknown_args = parser.parse_known_args()
    slurm_args = create_slurm_args(args, unknown_args, args.slurm_config_folder)
    main(args, slurm_args)
