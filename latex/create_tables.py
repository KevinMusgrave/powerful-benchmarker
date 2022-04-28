import argparse
import os
import sys

import pandas as pd
from pytorch_adapt.utils import common_functions as c_f

sys.path.insert(0, ".")
from latex import utils as latex_utils
from validator_tests.utils import utils
from validator_tests.utils.constants import add_exp_group_args
from validator_tests.utils.df_utils import get_name_from_exp_groups


def save_to_latex(df, folder, filename):
    c_f.makedir_if_not_there(folder)
    df_style = df.style.highlight_max(props="textbf:--rwrap")
    latex_str = df_style.format(escape="latex", na_rep="-",).to_latex(
        hrules=True,
        position_float="centering",
    )
    full_path = os.path.join(folder, f"{filename}.txt")
    with open(full_path, "w") as text_file:
        text_file.write(latex_str)


def preprocess_df(df):
    latex_utils.convert_adapter_name(df)
    return df


def remove_accuracy_name_multiindex(df):
    accuracy_name = df.columns.levels[0]
    assert len(accuracy_name) == 1
    accuracy_name = accuracy_name[0]
    df = df.droplevel(0, axis=1)
    return df, accuracy_name


def postprocess_df(df):
    df = pd.concat(df, axis=0)
    print(df)
    df = df.pivot(index="adapter", columns="task")
    df, accuracy_name = remove_accuracy_name_multiindex(df)
    df = latex_utils.add_source_only(df, accuracy_name)
    df = latex_utils.shortened_task_names(df)
    df = (df * 100).round(1)
    return df


def main(args):
    exp_groups = utils.get_exp_groups(args, exp_folder=args.input_folder)
    df = []
    for e in exp_groups:
        filename = os.path.join(args.input_folder, e, f"{args.filename}.csv")
        curr_df = pd.read_csv(filename)
        curr_df = preprocess_df(curr_df)
        df.append(curr_df)

    df = postprocess_df(df)
    print(df)
    output_folder = os.path.join(
        args.output_folder, get_name_from_exp_groups(exp_groups)
    )
    save_to_latex(df, output_folder, args.filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_exp_group_args(parser)
    parser.add_argument("--input_folder", type=str, default="tables")
    parser.add_argument("--output_folder", type=str, default="tables_latex")
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()
    main(args)
