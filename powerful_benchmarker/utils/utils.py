import os

import yaml


def create_slurm_args(args, other_args, folder):
    slurm_config_file = os.path.join(
        folder, "slurm_configs", f"{args.slurm_config}.yaml"
    )

    with open(slurm_config_file, "r") as f:
        slurm_args = yaml.safe_load(f)

    for s in other_args:
        if s == "":
            continue
        k, v = s.split("=")
        slurm_args[k.lstrip("--")] = v

    return slurm_args
