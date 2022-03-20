import argparse
import copy
import os
import shutil
import sys
from functools import partialmethod

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from powerful_benchmarker.utils.utils import convert_unknown_args
from validator_tests import configs
from validator_tests.utils import utils
from validator_tests.utils.constants import VALIDATOR_TESTS_FOLDER

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_curr_dict(curr_dict):
    x = set(curr_dict.keys()).intersection({"validator", "validator_args", "score"})
    if len(x) > 0:
        raise KeyError("curr_dict already has some validation related keys")


def save_df(validator_name, validator_args_str, all_scores):
    def fn(folder):
        df = pd.DataFrame(all_scores)
        filepath = utils.get_df_filepath(folder, validator_name, validator_args_str)
        df.to_pickle(filepath)
        all_scores.clear()

    return fn


def get_and_save_scores(validator_name, validator, validator_args_str, all_scores):
    def fn(epoch, x, exp_config, exp_folder):
        if isinstance(validator, configs.DEV):
            temp_folder = os.path.join(
                exp_folder, VALIDATOR_TESTS_FOLDER, f"DEV_{validator.layer}"
            )
            validator.validator.temp_folder = temp_folder
            if os.path.isdir(temp_folder):
                shutil.rmtree(temp_folder)  # delete any old copies
        score = validator.score(x, exp_config, DEVICE)
        curr_dict = copy.deepcopy(exp_config)
        assert_curr_dict(curr_dict)
        curr_dict["trial_params"] = utils.dict_to_str(curr_dict["trial_params"])
        curr_dict["epoch"] = epoch
        curr_dict.update(
            {
                "validator": validator_name,
                "validator_args": validator_args_str,
                "score": score,
            }
        )
        all_scores.append(curr_dict)

    return fn


def get_validator_and_condition_fn(
    validator_name, validator_args, trial_range, exp_folder, exp_group, exp_name
):
    validator = getattr(configs, validator_name)(validator_args)
    validator_args_str = utils.dict_to_str(validator.validator_args)
    exp_folders = utils.get_exp_folders(os.path.join(exp_folder, exp_group), exp_name)
    condition_fn = utils.get_condition_fn(
        validator_name, validator_args_str, trial_range
    )
    return validator, validator_args_str, exp_folders, condition_fn


def main(args, validator_args):
    (
        validator,
        validator_args_str,
        exp_folders,
        condition_fn,
    ) = get_validator_and_condition_fn(
        args.validator,
        validator_args,
        args.trial_range,
        args.exp_folder,
        args.exp_group,
        args.exp_name,
    )
    all_scores = []
    fn = get_and_save_scores(args.validator, validator, validator_args_str, all_scores)
    end_fn = save_df(args.validator, validator_args_str, all_scores)
    utils.apply_to_data(exp_folders, condition_fn, fn, end_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--exp_group", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--validator", type=str, required=True)
    parser.add_argument("--trial_range", nargs="+", type=int, default=[])
    args, unknown_args = parser.parse_known_args()
    validator_args = convert_unknown_args(unknown_args)
    main(args, validator_args)
