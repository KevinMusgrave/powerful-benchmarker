import json
import os
import pathlib
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


def append_jobid_to_file(jobid, jobname, filename):
    print(f"running job_id = {jobid}")
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            jobids = json.load(f)
    else:
        jobids = {}
    jobids[jobid] = jobname
    with open(filename, "w") as f:
        json.dump(jobids, f, indent=2)


def kill_all_jobs(exp_folder, jobids_file):
    all_jobids_filename = os.path.join(exp_folder, jobids_file)
    if not os.path.isfile(all_jobids_filename):
        print(f"{all_jobids_filename} file not found, skipping")
        return

    filetype = pathlib.Path(all_jobids_filename).suffix

    if filetype == ".json":
        with open(all_jobids_filename, "r") as f:
            jobids = json.load(f)
        jobids = " ".join(list(jobids.keys()))

    else:
        with open(all_jobids_filename, "r") as f:
            jobids = " ".join([line.rstrip("\n") for line in f])

    command = f"scancel {jobids}"
    print("killing slurm jobs")
    subprocess.run(command.split(" "))
    print(f"deleting {jobids_file}")
    os.remove(all_jobids_filename)


def jobs_that_are_still_running(exp_folder, jobids_file):
    x = subprocess.run("squeue --nohead --format %F".split(" "), capture_output=True)
    jobid_list = x.stdout.decode("utf-8").split("\n")

    all_jobids_filename = os.path.join(exp_folder, jobids_file)

    if os.path.isfile(all_jobids_filename):
        with open(all_jobids_filename, "r") as f:
            y = list(json.load(f).keys())

        return set(y).intersection(jobid_list)
    return {}


def create_exp_group_name(
    dataset,
    src_domains,
    target_domains,
    feature_layers,
    optimizers,
    lr_multipliers,
    validator=None,
):
    src_domains = "_".join(src_domains)
    target_domains = "_".join(target_domains)
    exp_group_name = f"{dataset}_{src_domains}_{target_domains}"
    if validator:
        exp_group_name += f"_{validator}"
    feature_layers = "".join(f"fl{str(x)}" for x in sorted(feature_layers))
    optimizers = "".join(x for x in sorted(optimizers))
    lr_multipliers = "".join(
        f"lr{f'{x:.1f}' if x < 1 else int(x)}" for x in sorted(lr_multipliers)
    )
    exp_group_name += f"_{feature_layers}_{optimizers}_{lr_multipliers}"
    return exp_group_name
