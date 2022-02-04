import argparse
import os
import subprocess
import sys

import submitit
import torch
import yaml

sys.path.insert(0, "src")
from powerful_benchmarker.utils.constants import BEST_TRIAL_FILENAME, add_default_args


def get_group_config(args):
    config_file = os.path.join("group_configs", f"{args.group_config}.yaml")
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def get_group_config_str(exp_folder, cfg):
    x = ""
    for k, v in cfg.items():
        if k == "fixed_param_source":
            x += f" --{k} {os.path.join(exp_folder, v)}"
        else:
            x += f" --{k}" if v is True else f" --{k} {v}"
    return x


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


def already_done(exp_folder, config_names):
    all_done = True
    for c in config_names:
        full_path = os.path.join(exp_folder, c)
        best_trial_file = os.path.join(full_path, BEST_TRIAL_FILENAME)
        if not os.path.isfile(best_trial_file):
            all_done = False
            break
    return all_done


def rotate(l, n):
    return l[n:] + l[:n]


def base_command(dataset_folder, exp_folder, exp_name, adapter, gcfg):
    x = f"python main.py --exp_folder {exp_folder} --exp_name {exp_name} --adapter {adapter} --dataset_folder {dataset_folder}"
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
    use_devices = ",".join([str(x) for x in rotate(gpu_list, local_rank)])
    command = base_command(cfg.dataset_folder, exp_folder, exp_name, config_name, gcfg)
    full_command = f"bash -i ./scripts/{cfg.script_wrapper} {exp_name} {str(cfg.script_wrapper_timeout)} {exp_folder} {cfg.conda_env} {use_devices} {BEST_TRIAL_FILENAME}".split(
        " "
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
    ]

    for i in [0, 1]:
        assert len(set(x[i] for x in exp_names)) == len(exp_names)

    exp_names = [(k, v) for k, v in exp_names if k in cfg.config_names]

    # make sure experiments are unique
    for i in [0, 1]:
        assert len(set(x[i] for x in exp_names)) == len(exp_names)

    gcfg = get_group_config(cfg)

    # catch the 0.1 case in a safe way
    if (gcfg["lr_multiplier"] - 1) < -0.5:
        lr_str = f"{gcfg['lr_multiplier']:.1f}"
    else:
        lr_str = f"{int(gcfg['lr_multiplier'])}"

    exp_group_name = f"{gcfg['dataset']}_{gcfg['src_domains']}_{gcfg['target_domains']}_{gcfg['validator']}_fl{gcfg['feature_layer']}_{gcfg['optimizer']}_lr{lr_str}"
    exp_folder = os.path.join(cfg.exp_folder, exp_group_name)

    if already_done(exp_folder, cfg.config_names):
        print("These experiments are already done. Exiting.")
        return

    num_tasks = len(exp_names)
    executor = submitit.AutoExecutor(folder=os.path.join(exp_folder, "slurm_logs"))
    slurm_args["job_name"] = f"{exp_group_name}_" + "_".join(cfg.config_names)
    executor.update_parameters(
        timeout_min=0,
        tasks_per_node=num_tasks,
        slurm_additional_parameters=slurm_args,
    )
    job = executor.submit(exp_launcher, cfg, exp_folder, exp_names, gcfg)
    jobid = job.job_id
    print(f"running job_id = {jobid}")
    all_jobids_filename = os.path.join(cfg.exp_folder, "all_jobids.txt")
    with open(all_jobids_filename, "a") as fd:
        fd.write(f"{jobid}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "dataset_folder", "conda_env"])
    parser.add_argument("--script_wrapper_timeout", type=int, default=1200)
    parser.add_argument("--config_names", nargs="+", type=str, required=True)
    parser.add_argument("--script_wrapper", type=str, default="script_wrapper.sh")
    parser.add_argument("--slurm_config", type=str, required=True)
    parser.add_argument("--group_config", type=str, required=True)
    args, unknown_args = parser.parse_known_args()

    slurm_args = create_slurm_args(args, unknown_args)
    main(args, slurm_args)
