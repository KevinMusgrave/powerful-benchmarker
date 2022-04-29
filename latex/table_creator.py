import os

import pandas as pd
from pytorch_adapt.utils import common_functions as c_f

from validator_tests.utils import utils
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


def table_creator(args, basename, preprocess_df, postprocess_df):
    exp_groups = utils.get_exp_groups(args, exp_folder=args.input_folder)
    df = []
    for e in exp_groups:
        filename = os.path.join(args.input_folder, e, f"{basename}.csv")
        curr_df = pd.read_csv(filename)
        curr_df = preprocess_df(curr_df)
        df.append(curr_df)

    df = postprocess_df(df)
    print(df)
    output_folder = os.path.join(
        args.output_folder, get_name_from_exp_groups(exp_groups)
    )
    save_to_latex(df, output_folder, basename)
