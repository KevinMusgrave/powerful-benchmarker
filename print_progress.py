import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import (
    BEST_TRIAL_FILENAME,
    JOBIDS_FILENAME,
    TRIALS_FILENAME,
    add_default_args,
)
from powerful_benchmarker.utils.utils import jobs_that_are_still_running
from validator_tests.utils.constants import JOBIDS_FILENAME as V_JOBSID_FILENAME
from validator_tests.utils.constants import VALIDATOR_TESTS_FOLDER


def num_jobs_running(exp_folder):
    x = jobs_that_are_still_running(exp_folder, JOBIDS_FILENAME)
    y = jobs_that_are_still_running(exp_folder, V_JOBSID_FILENAME)
    return f"{len(x)} algorithm jobs and {len(y)} validator jobs still running\n"


def update_validator_progress_dicts(exp_name, contents, val, val_details):
    curr_val = validator_test_progress(contents)
    curr_details = {}
    for k, v in curr_val.items():
        val[k] += v
        x = str(v)
        if v < 100:
            x += "    Validator Not Done"
        curr_details[k] = x
    val_details[exp_name] = curr_details


def validator_test_progress(contents):
    output = defaultdict(int)
    for x in contents:
        contents = glob.glob(os.path.join(x, VALIDATOR_TESTS_FOLDER, "*.pkl"))
        for pkl_file in contents:
            output[Path(pkl_file).stem] += 1
    return output


def is_done(e):
    best_trial_file = os.path.join(e, BEST_TRIAL_FILENAME)
    return os.path.isfile(best_trial_file)


def read_trials_csv(e):
    filepath = os.path.join(e, TRIALS_FILENAME)
    if os.path.isfile(filepath):
        trials = pd.read_csv(filepath)
        num_success = len(trials[trials["state"] == "COMPLETE"])
        return f"{num_success} / {len(trials)}"
    return "0 / 0"


def count_exp_folders(contents):
    num_folders = 0
    for x in contents:
        basename = os.path.basename(x)
        if os.path.isdir(x) and (
            basename.isdigit() or basename.startswith("reproduction")
        ):
            num_folders += 1
    return num_folders


def progress(cfg, exps):
    folder_progress = {}
    validator_progress = defaultdict(int)
    validator_progress_details = {}
    for e in exps:
        e = os.path.normpath(e)
        exp_name = os.path.basename(e)
        if not os.path.isdir(e) or exp_name == cfg.slurm_folder:
            continue
        contents = glob.glob(f"{e}/*")
        num_folders = count_exp_folders(contents)
        num_success_str = read_trials_csv(e)
        output_str = f"{num_folders} folders:   {num_success_str}"
        if not is_done(e):
            output_str += "    In Progress"
        folder_progress[exp_name] = output_str
        if cfg.with_validator_progress:
            update_validator_progress_dicts(
                exp_name, contents, validator_progress, validator_progress_details
            )
    return folder_progress, validator_progress, validator_progress_details


def main(cfg):
    experiment_paths = sorted(glob.glob(f"{cfg.exp_folder}/*"))
    all_folders = {}
    for p in experiment_paths:
        if os.path.isdir(p):
            exps = sorted(glob.glob(f"{p}/*"))
            folder_progress, validator_progress, validator_progress_details = progress(
                cfg, exps
            )
            curr_dict = {
                "folder_progress": folder_progress,
            }
            if cfg.with_validator_progress:
                curr_dict.update(
                    {
                        "validator_progress": validator_progress,
                        "validator_progress_details": validator_progress_details,
                    }
                )
            all_folders[os.path.basename(p)] = curr_dict

    out_string = num_jobs_running(cfg.exp_folder)
    out_string += json.dumps(all_folders, indent=4, sort_keys=True)
    if cfg.save_to_file:
        with open(cfg.save_to_file, "w") as f:
            f.write(out_string)
    else:
        print(out_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "slurm_folder"])
    parser.add_argument(
        "--save_to_file",
        type=str,
        default=None,
    )
    parser.add_argument("--with_validator_progress", action="store_true")
    args = parser.parse_args()
    main(args)
