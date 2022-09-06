import pandas as pd

from latex.correlation import base_filename, get_preprocess_df
from latex.correlation_bar_plot import reshape_and_plot
from latex.correlation_single_adapter import get_postprocess_df
from latex.table_creator import table_creator


def postprocess_df_wrapper(df):
    dfs = get_postprocess_df(remove_index_names=False)(df)
    for k, df in dfs.items():
        dfs[k] = pd.concat([df], keys=[k], names=["adapter"])
    return pd.concat(list(dfs.values()), axis=0)


def new_col_fn(df):
    new_col = df.apply(
        lambda x: f'{x["adapter"]} / {x["validator"]}: {x["validator_args"]}', axis=1
    )
    return df.assign(validator_as_str=new_col).drop(
        columns=["adapter", "validator", "validator_args"]
    )


def correlation_bar_plot_adapter_validator_pairs(args, name, src_threshold):
    basename = base_filename(name, True, src_threshold)

    df, output_folder = table_creator(
        args,
        basename,
        preprocess_df=get_preprocess_df(per_adapter=True),
        postprocess_df=postprocess_df_wrapper,
        do_save_to_latex=False,
    )

    basename = basename.replace("_per_adapter", "")
    basename = f"{basename}_adapter_validator_pairs"
    reshape_and_plot(
        df,
        output_folder,
        basename,
        name,
        new_col_fn,
        figsize=(12, 72),
        y_tick_labelsize=6,
    )
