import argparse
import json
import os

import h5py
import pandas as pd
import torch

from powerful_benchmarker.utils.constants import add_default_args
from validator_tests import configs
from validator_tests.utils.constants import get_all_dfs
from validator_tests.utils.df_utils import print_validators_with_nan

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    pd.options.display.max_colwidth = 100
    exp_folder = os.path.join(args.exp_folder, args.exp_group)
    df = get_all_dfs(exp_folder)
    df = print_validators_with_nan(df, return_df=True)
    validator_name = "MMDPerClassFixedB"
    df = df[df["validator"] == validator_name]
    df = df.iloc[0]

    validator = getattr(configs, validator_name)(json.loads(df["validator_args"]))
    if validator_name == "DEV":
        validator.validator.temp_folder = "DEV_temp_folder"

    features_file = os.path.join(
        exp_folder, df["exp_name"], str(df["trial_num"]), "features", "features.hdf5"
    )
    with h5py.File(features_file, "r") as data:
        epoch_data = data[f"{df['epoch']}"]
        score = validator.score(epoch_data, None, DEVICE)

    print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--exp_group", type=str, required=True)
    args = parser.parse_args()
    main(args)


# MMD and MMDPerClass NaNs are caused by collapsed embeddings.
# The distance between the embeddings is 0, so the median scaling (1/median(x)) results in inf

# DEV NaN is also caused by collapsed embeddings.
# The softmaxed output of the discriminator end up being entirely for one class
# so the weights calculation becomes 1/0

# MMDPerClassFixedB NaNs are caused by either the source or target domain
# having only 1 embedding, i.e. the shape of src_features or target_features is
# (1, feature_size). Since self-distances are excluded, the denominator
# used for normalization is 0. See denom calculation in
# pytorch_adapt.layers.utils.get_mmd_quadratic_batched
