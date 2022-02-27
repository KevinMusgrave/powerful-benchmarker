import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "src")
from powerful_benchmarker.utils.constants import BEST_TRIAL_FILENAME, add_default_args


def is_done(e):
    best_trial_file = os.path.join(e, BEST_TRIAL_FILENAME)
    return os.path.isfile(best_trial_file)


def count_exp_folders(contents):
    num_folders = 0
    for x in contents:
        basename = os.path.basename(x)
        if os.path.isdir(x) and (
            basename.isdigit() or basename.startswith("reproduction")
        ):
            num_folders += 1
    return num_folders


def print_folder_progress(cfg, exps):
    output = {}
    for e in exps:
        e = os.path.normpath(e)
        contents = glob.glob(f"{e}/*")
        num_folders = count_exp_folders(contents)
        output_str = f"{num_folders}"
        if not is_done(e):
            output_str += ", In Progress"
        output[os.path.basename(e)] = output_str
    return output


def main(cfg):
    experiment_paths = sorted(glob.glob(f"{cfg.exp_folder}/*"))
    all_folders = {}
    for p in experiment_paths:
        if os.path.isdir(p):
            exps = sorted(glob.glob(f"{p}/*"))
            all_folders[os.path.basename(p)] = print_folder_progress(cfg, exps)

    out_string = json.dumps(all_folders, indent=4)
    if cfg.save_to_file:
        with open(cfg.save_to_file, "w") as f:
            f.write(out_string)
    else:
        print(out_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument(
        "--save_to_file",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args)
