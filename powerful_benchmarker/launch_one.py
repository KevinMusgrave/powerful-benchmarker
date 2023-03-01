import argparse
import os
import subprocess
import sys

import submitit
import torch
import yaml

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import (
    BEST_TRIAL_FILENAME,
    JOBIDS_FILENAME,
    add_default_args,
)
from powerful_benchmarker.utils.utils import (
    append_jobid_to_file,
    create_exp_group_name,
    create_slurm_args,
    get_yaml_config_folder,
    get_yaml_config_path,
    rotate,
)


def get_group_config(args):
    x = {}
    for g in args.group_configs:
        config_file = get_yaml_config_path("group_configs", g)
        with open(config_file, "r") as f:
            x.update(yaml.safe_load(f))
    # required args that can be in either configs or command line
    for k in ["src_domains", "target_domains"]:
        if k not in x:
            x[k] = getattr(args, k)
        assert isinstance(x[k], list) and len(x[k]) > 0
    return x


def get_group_config_str(exp_folder, cfg):
    x = ""
    for k, v in cfg.items():
        if k in ["src_domains", "target_domains"]:
            list_str = " ".join(v)
            x += f" --{k} {list_str}"
        else:
            x += f" --{k}" if v is True else f" --{k} {v}"
    return x


def already_done(exp_folder, config_names):
    all_done = True
    for c in config_names:
        full_path = os.path.join(exp_folder, c)
        best_trial_file = os.path.join(full_path, BEST_TRIAL_FILENAME)
        if not os.path.isfile(best_trial_file):
            all_done = False
            break
    return all_done


def base_command(dataset_folder, exp_folder, exp_name, adapter, gcfg):
    x = f"python powerful_benchmarker/main.py --exp_folder {exp_folder} --exp_name {exp_name} --adapter {adapter} --dataset_folder {dataset_folder}"
    gcfg_str = get_group_config_str(exp_folder, gcfg)
    x += gcfg_str
    return x


def exp_launcher(cfg, exp_folder, exp_names, gcfg):
    num_gpus = torch.cuda.device_count()
    print("num gpus available in exp_launcher =", num_gpus)

    job_env = submitit.JobEnvironment()
    local_rank = job_env.local_rank
    exp_name, config_name = exp_names[local_rank]
    gpu_list = list(range(num_gpus))
    use_devices = ",".join(str(x) for x in rotate(gpu_list, local_rank))
    command = base_command(cfg.dataset_folder, exp_folder, exp_name, config_name, gcfg)
    args = f"{exp_name} {str(cfg.script_wrapper_timeout)} {exp_folder} {cfg.conda_env} {use_devices} {BEST_TRIAL_FILENAME}"
    full_command = (
        f"bash -i ./powerful_benchmarker/scripts/script_wrapper.sh {args}".split(" ")
    )
    full_command += [command]
    subprocess.run(full_command)


def main(cfg, slurm_args):
    exp_names = [
        ("cdan", "CDANConfig"),
        ("dann", "DANNConfig"),
        ("mcc", "MCCConfig"),
        ("mcd", "MCDConfig"),
        ("mmd", "MMDConfig"),
        ("bsp", "BSPConfig"),
        ("bnm", "BNMConfig"),
        ("gvb", "GVBConfig"),
        ("atdoc", "ATDOCConfig"),
        ("im", "IMConfig"),
        ("epoch_0", "PretrainerConfig")
    ]

    for i in [0, 1]:
        assert len(set(x[i] for x in exp_names)) == len(exp_names)

    exp_names.append(("trial_run", "DANNConfig"))  # special name for testing
    exp_names = [(k, v) for k, v in exp_names if k in cfg.config_names]

    # make sure experiments are unique
    for i in [0, 1]:
        assert len(set(x[i] for x in exp_names)) == len(exp_names)

    gcfg = get_group_config(cfg)
    if cfg.is_stress_test:
        exp_group_name = "stress_test"
    else:
        exp_group_name = create_exp_group_name(
            gcfg["dataset"],
            gcfg["src_domains"],
            gcfg["target_domains"],
            [gcfg["feature_layer"]],
            [gcfg["optimizer"]],
            [gcfg["lr_multiplier"]],
            gcfg.get("validator"),
        )
    exp_folder = os.path.join(cfg.exp_folder, exp_group_name)

    if already_done(exp_folder, cfg.config_names):
        print("These experiments are already done. Exiting.")
        return

    num_tasks = len(exp_names)
    executor = submitit.AutoExecutor(folder=os.path.join(exp_folder, cfg.slurm_folder))
    if cfg.is_stress_test:
        job_name = exp_group_name
    else:
        job_name = f"{exp_group_name}_" + "_".join(cfg.config_names)
    slurm_args["job_name"] = job_name
    executor.update_parameters(
        timeout_min=0,
        tasks_per_node=num_tasks,
        slurm_additional_parameters=slurm_args,
    )
    job = executor.submit(exp_launcher, cfg, exp_folder, exp_names, gcfg)
    all_jobids_filename = os.path.join(cfg.exp_folder, JOBIDS_FILENAME)
    append_jobid_to_file(job.job_id, job_name, all_jobids_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(
        parser, ["exp_folder", "dataset_folder", "conda_env", "slurm_folder"]
    )
    parser.add_argument("--is_stress_test", action="store_true")
    parser.add_argument("--script_wrapper_timeout", type=int, default=1200)
    parser.add_argument("--config_names", nargs="+", type=str, required=True)
    parser.add_argument("--slurm_config", type=str, required=True)
    parser.add_argument("--group_configs", nargs="+", type=str, required=True)

    # can be specified in group config or here
    parser.add_argument("--src_domains", nargs="+", type=str)
    parser.add_argument("--target_domains", nargs="+", type=str)
    args, unknown_args = parser.parse_known_args()

    slurm_args = create_slurm_args(args, unknown_args, get_yaml_config_folder())
    main(args, slurm_args)
