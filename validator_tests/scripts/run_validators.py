import argparse
import math
import os
import subprocess
import sys

import numpy as np
import submitit
import torch
import yaml

sys.path.insert(0, "src")
from powerful_benchmarker.utils.constants import add_default_args

from . import flags as flags_module


def create_slurm_args(args, other_args):
    slurm_config_file = os.path.join("slurm_configs", f"{args.slurm_config}.yaml")

    with open(slurm_config_file, "r") as f:
        slurm_args = yaml.safe_load(f)

    for s in unknown_args:
        if s == "":
            continue
        k, v = s.split("=")
        slurm_args[k.lstrip("--")] = v

    return slurm_args


def get_trial_range(to_run, trials_per_exp, exp_per_slurm_job):
    output = []
    num_trials = 100
    trial_nums = np.array_split(np.arange(num_trials), int(100 / trials_per_exp))
    for x in to_run:
        for y in trial_nums:
            y = f"{min(y)} {max(y)+1}"
            output.append(f"{x} --trial_range {y}")
    return np.array_split(np.array(output), math.ceil(len(output) / exp_per_slurm_job))


def rotate(l, n):
    return l[n:] + l[:n]


def exp_launcher(args, commands):
    num_gpus = torch.cuda.device_count()
    print("num gpus available in exp_launcher =", num_gpus)
    job_env = submitit.JobEnvironment()
    local_rank = job_env.local_rank
    gpu_list = list(range(num_gpus))
    use_devices = ",".join([str(x) for x in rotate(gpu_list, local_rank)])
    job_env = submitit.JobEnvironment()
    command = commands[job_env.local_rank]
    full_command = (
        f"bash -i ./scripts/script_wrapper.sh {args.conda_env} {use_devices}".split(" ")
    )
    full_command += [command]
    subprocess.run(full_command)


def run_slurm_job(args, slurm_args, commands):
    executor = submitit.AutoExecutor(
        folder=os.path.join(args.exp_folder, args.exp_group, "slurm_logs")
    )
    slurm_args["job_name"] = f"{args.flags}_validator_tests"
    executor.update_parameters(
        timeout_min=0,
        tasks_per_node=len(commands),
        slurm_additional_parameters=slurm_args,
    )
    job = executor.submit(exp_launcher, args, commands)
    jobid = job.job_id
    print(f"running job_id = {jobid}")
    all_jobids_filename = os.path.join(args.exp_folder, "all_jobids.txt")
    with open(all_jobids_filename, "a") as fd:
        fd.write(f"{jobid}\n")


def main(args, slurm_args):
    to_run = []
    for exp_name in args.exp_names:
        base_command = f"python main.py --exp_folder {args.exp_folder} --exp_group {args.exp_group} --exp_name {exp_name}"
        flags = getattr(flags_module, args.flags)()
        commands = [f"{base_command} {x}" for x in flags]
        to_run.extend(commands)

    to_run = get_trial_range(to_run, args.trials_per_exp, args.exp_per_slurm_job)
    print(f"{len(to_run)} slurm jobs")
    for commands in to_run:
        print(f"{len(commands)} exps in this job")
        if len(commands) > 1 and args.run:
            run_slurm_job(args, slurm_args, commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "conda_env"])
    parser.add_argument("--exp_group", type=str, required=True)
    parser.add_argument("--exp_names", nargs="+", type=str, required=True)
    parser.add_argument("--flags", type=str, required=True)
    parser.add_argument("--trials_per_exp", type=int, required=True)
    parser.add_argument("--exp_per_slurm_job", type=int, required=True)
    parser.add_argument("--slurm_config", type=str, required=True)
    parser.add_argument("--run", action="store_true")
    args, unknown_args = parser.parse_known_args()
    slurm_args = create_slurm_args(args, unknown_args)
    main(args, slurm_args)
