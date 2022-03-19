import os
import subprocess

import yaml


def convert_unknown_args(unknown_args):
    args = {}
    for s in unknown_args:
        if s == "":
            continue
        k, v = s.split("=")
        args[k.lstrip("--")] = v
    return args


def create_slurm_args(args, other_args, folder):
    slurm_config_file = os.path.join(
        folder, "slurm_configs", f"{args.slurm_config}.yaml"
    )

    with open(slurm_config_file, "r") as f:
        slurm_args = yaml.safe_load(f)

    slurm_args.update(convert_unknown_args(other_args))

    return slurm_args


def rotate(l, n):
    return l[n:] + l[:n]


def get_yaml_config_folder():
    return os.path.join("powerful_benchmarker", "yaml_configs")


def get_yaml_config_path(category, name):
    return os.path.join(get_yaml_config_folder(), category, f"{name}.yaml")


def append_jobid_to_file(jobid, filename):
    print(f"running job_id = {jobid}")
    with open(filename, "a") as fd:
        fd.write(f"{jobid}\n")


def kill_all_jobs(exp_folder, jobids_file):
    all_jobids_filename = os.path.join(exp_folder, jobids_file)
    with open(all_jobids_filename, "r") as f:
        jobids = " ".join([line.rstrip("\n") for line in f])

    command = f"scancel {jobids}"
    print("killing slurm jobs")
    subprocess.run(command.split(" "))
    print(f"deleting {jobids_file}")
    os.remove(all_jobids_filename)
