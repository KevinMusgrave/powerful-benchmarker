import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from validator_tests.utils import utils
from validator_tests.utils.constants import add_exp_group_args


def convert_adapter_name(df):
    df["adapter"] = df["adapter"].str.replace("Config", "")


def assign_shortened_task_name(df):
    task_name_map = {
        "mnist_mnist_mnistm": "MM",
        "officehome_art_clipart": "AC",
        "officehome_art_product": "AP",
        "officehome_art_real": "AR",
        "officehome_clipart_art": "CA",
        "officehome_clipart_product": "CP",
        "officehome_clipart_real": "CR",
        "officehome_product_art": "PA",
        "officehome_product_clipart": "PC",
        "officehome_product_real": "PR",
        "officehome_real_art": "RA",
        "officehome_real_clipart": "RC",
        "officehome_real_product": "RP",
    }

    new_col = df.apply(lambda x: task_name_map[x["task"]], axis=1)
    return df.assign(task=new_col)


def main(args):
    exp_groups = utils.get_exp_groups(args, exp_folder=args.input_folder)
    df = []
    for e in exp_groups:
        filename = os.path.join(args.input_folder, e, f"{args.filename}.csv")
        curr_df = pd.read_csv(filename)
        convert_adapter_name(curr_df)
        curr_df = assign_shortened_task_name(curr_df)
        df.append(curr_df)

    df = pd.concat(df, axis=1)
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_exp_group_args(parser)
    parser.add_argument("--input_folder", type=str, default="tables")
    parser.add_argument("--output_folder", type=str, default="latex_tables")
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()
    main(args)
