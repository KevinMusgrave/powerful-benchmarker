import argparse
import os
import subprocess

import yaml


def main(cfg, slurm_args):
    config_file = os.path.join(cfg.config_folder, f"{cfg.config}.yaml")

    with open(config_file, "r") as f:
        commands = yaml.safe_load(f)["commands"]

    for i, cs in enumerate(commands):
        c = " ".join(cs)
        c = f"{c} {slurm_args}"
        print(f"launching {c}")
        subprocess.run(c.split(" "))


if __name__ == "__main__":
    with open("constants.yaml", "r") as f:
        constants = yaml.safe_load(f)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config-folder", type=str, default="slurm_configs")
    parser.add_argument("--config", type=str, required=True)
    args, unknown_args = parser.parse_known_args()
    slurm_args = " ".join(unknown_args)
    main(args, slurm_args)
