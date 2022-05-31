import os

import pandas as pd
from pytorch_adapt.utils import common_functions as c_f

from latex.color_map_tags import create_color_map_tags, get_tags_dict
from latex.utils import resizebox
from validator_tests.utils import utils
from validator_tests.utils.df_utils import get_name_from_exp_groups


def save_to_latex(
    df,
    folder,
    filename,
    color_map_tag_kwargs,
    add_resizebox,
    highlight_max=True,
    highlight_min=False,
    highlight_max_subset=None,
    highlight_min_subset=None,
    final_str_hook=None,
    **kwargs,
):
    c_f.makedir_if_not_there(folder)
    tags_dict, color_map_tags = None, ""
    if color_map_tag_kwargs:
        color_map_tags = create_color_map_tags(df, **color_map_tag_kwargs)
        tags_dict = get_tags_dict(color_map_tag_kwargs["tag_prefix"], df.columns.values)

    df_style = df.style
    if highlight_max:
        df_style = df_style.highlight_max(
            subset=highlight_max_subset, props="textbf:--rwrap"
        )
    if highlight_min:
        df_style = df_style.highlight_min(
            subset=highlight_min_subset, props="textbf:--rwrap"
        )
    latex_str = df_style.format(
        tags_dict,
        escape="latex",
        na_rep="-",
    ).to_latex(hrules=True, position_float="centering", **kwargs)
    full_path = os.path.join(folder, f"{filename}.tex")

    if color_map_tags:
        newlines = "\n" * 10
        latex_str = f"{color_map_tags}{newlines}{latex_str}"
    if add_resizebox:
        latex_str = resizebox(latex_str)
    if final_str_hook:
        latex_str = final_str_hook(latex_str)
    with open(full_path, "w") as text_file:
        text_file.write(latex_str)


def table_creator(
    args,
    basename,
    preprocess_df,
    postprocess_df,
    color_map_tag_kwargs=None,
    add_resizebox=False,
    do_save_to_latex=True,
    caption_hook=None,
    **kwargs,
):
    exp_groups = utils.get_exp_groups(args, exp_folder=args.input_folder)
    df = []
    for e in exp_groups:
        filename = os.path.join(args.input_folder, e, f"{basename}.pkl")
        curr_df = pd.read_pickle(filename)
        curr_df = preprocess_df(curr_df)
        df.append(curr_df)

    df = postprocess_df(df)
    output_folder = os.path.join(
        args.output_folder, get_name_from_exp_groups(exp_groups)
    )
    if do_save_to_latex:
        if isinstance(df, dict):
            original_caption = kwargs.pop("caption", None)
            for k, x in df.items():
                curr_basename = f"{basename}_{k}"
                if caption_hook:
                    caption = caption_hook(original_caption, k)
                save_to_latex(
                    x,
                    output_folder,
                    curr_basename,
                    color_map_tag_kwargs,
                    add_resizebox,
                    label=curr_basename,
                    caption=caption,
                    **kwargs,
                )
        else:
            save_to_latex(
                df,
                output_folder,
                basename,
                color_map_tag_kwargs,
                add_resizebox,
                label=basename,
                **kwargs,
            )
    else:
        return df, output_folder
