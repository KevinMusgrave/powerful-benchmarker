import argparse
import os
import subprocess
import sys

import submitit
import torch
import yaml

sys.path.insert(0, "src")
from powerful_benchmarker.utils.constants import BEST_TRIAL_FILENAME


def already_done(experiment_path, config_names):
    all_done = True
    for c in config_names:
        full_path = os.path.join(experiment_path, c)
        best_trial_file = os.path.join(full_path, BEST_TRIAL_FILENAME)
        if not os.path.isfile(best_trial_file):
            all_done = False
            break
    return all_done


def rotate(l, n):
    return l[n:] + l[:n]


def base_command(experiment_name, adapter, experiment_path, cfg):
    x = f"python main.py --experiment_name {experiment_name} --adapter {adapter} --start_with_pretrained \
--num_workers {cfg.num_workers} --experiment_path {experiment_path} --dataset_folder {cfg.dataset_folder} \
--num_trials 100 --n_startup_trials 100 --batch_size 64 --max_epochs {cfg.max_epochs} --patience {cfg.patience} --validation_interval {cfg.validation_interval} \
--dataset {cfg.dataset} --src_domains {cfg.src_domain} --target_domains {cfg.target_domain} --validator {cfg.validator} \
--feature_layer {cfg.feature_layer} --optimizer_name {cfg.optimizer_name} --lr_multiplier {cfg.lr_multiplier} --num_reproduce 4"
    if cfg.fixed_param_source:
        fps = os.path.join(experiment_path, cfg.fixed_param_source)
        x += f" --fixed_param_source {fps}"
    for name in ["save_features", "download_datasets"]:
        if getattr(cfg, name):
            x += f" --{name}"
    return x


def exp_launcher(cfg, experiment_path, exp_names):
    num_gpus = torch.cuda.device_count()
    print("num gpus available in exp_launcher =", num_gpus)

    job_env = submitit.JobEnvironment()
    local_rank = job_env.local_rank
    dir_name, config_name = exp_names[local_rank]
    gpu_list = list(range(num_gpus))
    use_devices = ",".join([str(x) for x in rotate(gpu_list, local_rank)])
    command = base_command(dir_name, config_name, experiment_path, cfg)
    full_command = f"bash -i ./scripts/{cfg.script_wrapper} {dir_name} {str(cfg.script_wrapper_timeout)} {experiment_path} {cfg.conda_env} {use_devices} {BEST_TRIAL_FILENAME}".split(
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

    # catch the 0.1 case in a safe way
    if (cfg.lr_multiplier - 1) < -0.5:
        lr_str = f"{cfg.lr_multiplier:.1f}"
    else:
        lr_str = f"{int(cfg.lr_multiplier)}"

    experiment_group_name = f"{cfg.dataset}_{cfg.src_domain}_{cfg.target_domain}_{cfg.validator}_fl{cfg.feature_layer}_{cfg.optimizer_name}_lr{lr_str}"
    experiment_path = os.path.join(cfg.root_experiment_folder, experiment_group_name)

    if already_done(experiment_path, cfg.config_names):
        print("These experiments are already done. Exiting.")
        return

    num_tasks = len(exp_names)
    executor = submitit.AutoExecutor(folder=os.path.join(experiment_path, "slurm_logs"))
    slurm_args["job_name"] = f"{experiment_group_name}_" + "_".join(cfg.config_names)
    executor.update_parameters(
        timeout_min=0,
        tasks_per_node=num_tasks,
        slurm_additional_parameters=slurm_args,
    )
    job = executor.submit(exp_launcher, cfg, experiment_path, exp_names)
    jobid = job.job_id
    print(f"running job_id = {jobid}")
    all_jobids_filename = os.path.join(cfg.root_experiment_folder, "all_jobids.txt")
    with open(all_jobids_filename, "a") as fd:
        fd.write(f"{jobid}\n")


if __name__ == "__main__":
    with open("constants.yaml", "r") as f:
        constants = yaml.safe_load(f)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--script_wrapper_timeout", type=int, default=1200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--root_experiment_folder",
        type=str,
        default=constants["experiment_folder"],
    )
    parser.add_argument(
        "--dataset_folder", type=str, default=constants["dataset_folder"]
    )
    parser.add_argument("--conda_env", type=str, default=constants["conda_env"])
    parser.add_argument("--config_names", nargs="+", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--src_domain", type=str, required=True)
    parser.add_argument("--target_domain", type=str, required=True)
    parser.add_argument("--validator", type=str, required=True)
    parser.add_argument("--feature_layer", type=int, required=True)
    parser.add_argument("--optimizer_name", type=str, required=True)
    parser.add_argument("--lr_multiplier", type=float, required=True)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--fixed_param_source", type=str, default=None)
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--download_datasets", action="store_true")
    parser.add_argument("--script_wrapper", type=str, default="script_wrapper.sh")
    args, unknown_args = parser.parse_known_args()

    slurm_args = {}
    for s in unknown_args:
        if s == "":
            continue
        k, v = s.split("=")
        slurm_args[k.lstrip("--")] = v

    main(args, slurm_args)
