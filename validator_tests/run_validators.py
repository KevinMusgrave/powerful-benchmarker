import argparse
import copy
import math
import os
import subprocess
import sys

import numpy as np
import submitit
import torch

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from powerful_benchmarker.utils.utils import (
    append_jobid_to_file,
    create_slurm_args,
    rotate,
)
from validator_tests import flags as flags_module
from validator_tests.main import get_validator_and_condition_fn
from validator_tests.utils.constants import JOBIDS_FILENAME
from validator_tests.utils.utils import apply_to_data, filter_exp_groups


def split_into_batches(to_run, exp_per_slurm_job):
    return np.array_split(np.array(to_run), math.ceil(len(to_run) / exp_per_slurm_job))


def get_trial_ranges(trials_per_exp):
    num_trials = 100
    trial_nums = np.array_split(np.arange(num_trials), int(100 / trials_per_exp))
    return [(min(y), max(y) + 1) for y in trial_nums]


def exp_launcher(args, commands):
    num_gpus = torch.cuda.device_count()
    print("num gpus available in exp_launcher =", num_gpus)
    job_env = submitit.JobEnvironment()
    local_rank = job_env.local_rank
    gpu_list = list(range(num_gpus))
    use_devices = ",".join([str(x) for x in rotate(gpu_list, local_rank)])
    job_env = submitit.JobEnvironment()
    command = commands[job_env.local_rank]
    full_command = f"bash -i ./validator_tests/scripts/script_wrapper.sh {args.conda_env} {use_devices}".split(
        " "
    )
    full_command += [command]
    subprocess.run(full_command)


def run_slurm_job(args, slurm_args, exp_group, commands):
    executor = submitit.AutoExecutor(
        folder=os.path.join(args.exp_folder, exp_group, "slurm_logs")
    )
    slurm_args["job_name"] = f"{args.flags}_validator_tests"
    executor.update_parameters(
        timeout_min=0,
        tasks_per_node=len(commands),
        slurm_additional_parameters=slurm_args,
    )
    job = executor.submit(exp_launcher, args, commands)
    jobid = job.job_id
    all_jobids_filename = os.path.join(args.exp_folder, JOBIDS_FILENAME)
    append_jobid_to_file(jobid, all_jobids_filename)


def flags_to_strs(flags):
    output = []
    for f in flags:
        trial_range = f.pop("trial_range")
        x = " ".join([f"--{k}={v}" for k, v in f.items()])
        x += f" --trial_range {trial_range[0]} {trial_range[1]}"
        output.append(x)
    return output


def get_count_fn(x):
    def fn(*args, **kwargs):
        x.append(True)

    return fn


def remove_completed_flags(flags, trial_ranges, exp_folder, exp_group, exp_name):
    print("removing completed flags")
    all_flags = []
    for f in flags:
        for t in trial_ranges:
            all_flags.append({**f, "trial_range": t})

    keep_flags = []
    for f in all_flags:
        f_copy = copy.deepcopy(f)
        validator_name = f_copy.pop("validator")
        trial_range = f_copy.pop("trial_range")
        validator, _, exp_folders, condition_fn = get_validator_and_condition_fn(
            validator_name, f_copy, trial_range, exp_folder, exp_group, exp_name
        )
        count_list = []
        end_fn = get_count_fn(count_list)
        apply_to_data(exp_folders, condition_fn, end_fn=end_fn)
        if len(count_list) > 0:
            keep_flags.append(f)
    print(f"kept {len(keep_flags)} flags")
    return keep_flags


def no_duplicates(x):
    return len(x) == len(set(x))


def launcher(args, slurm_args, exp_groups):
    to_run = []
    assert no_duplicates(exp_groups)
    assert no_duplicates(args.exp_names)
    for exp_group in exp_groups:
        for exp_name in args.exp_names:
            print(f"creating flags for {exp_group}/{exp_name}/{args.flags}")
            base_command = f"python validator_tests/main.py --exp_folder {args.exp_folder} --exp_group {exp_group} --exp_name {exp_name}"
            flags = getattr(flags_module, args.flags)()
            trial_ranges = get_trial_ranges(args.trials_per_exp)
            flags = remove_completed_flags(
                flags, trial_ranges, args.exp_folder, exp_group, exp_name
            )
            flags = flags_to_strs(flags)
            commands = [f"{base_command} {x}" for x in flags]
            to_run.extend(commands)

    if len(to_run) == 0:
        print("Jobs are already done. Exiting.")
        return

    to_run = split_into_batches(to_run, args.exp_per_slurm_job)
    print(f"{len(to_run)} slurm jobs")
    for commands in to_run:
        print(f"{len(commands)} exps in this job")
        if len(commands) > 0 and args.run:
            run_slurm_job(args, slurm_args, exp_group, commands)


def main(args, slurm_args):
    if args.exp_groups:
        exp_groups = args.exp_groups
    elif args.exp_group_prefix:
        exp_groups = filter_exp_groups(
            args.exp_folder,
            prefix=args.exp_group_prefix,
            suffix=args.exp_group_suffix,
            contains=args.exp_group_contains,
        )
    else:
        raise ValueError("exp_groups or exp_group_prefix must be specified")

    launcher(args, slurm_args, exp_groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "conda_env"])
    parser.add_argument("--exp_groups", nargs="+", type=str)
    parser.add_argument("--exp_group_prefix", type=str)
    parser.add_argument("--exp_group_suffix", type=str)
    parser.add_argument("--exp_group_contains", type=str)
    parser.add_argument("--exp_names", nargs="+", type=str, required=True)
    parser.add_argument("--flags", type=str, required=True)
    parser.add_argument("--trials_per_exp", type=int, required=True)
    parser.add_argument("--exp_per_slurm_job", type=int, required=True)
    parser.add_argument("--slurm_config", type=str, required=True)
    parser.add_argument("--run", action="store_true")
    args, unknown_args = parser.parse_known_args()
    slurm_args = create_slurm_args(args, unknown_args, "validator_tests")
    main(args, slurm_args)
