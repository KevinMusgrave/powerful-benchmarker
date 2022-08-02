from latex.correlation import base_filename, get_preprocess_df
from latex.correlation_bar_plot import reshape_and_plot
from latex.correlation_single_adapter import postprocess_df
from latex.table_creator import table_creator


def correlation_bar_plot_single_adapter(args, name, src_threshold):
    basename = base_filename(name, True, src_threshold)

    dfs, output_folder = table_creator(
        args,
        basename,
        preprocess_df=get_preprocess_df(per_adapter=True),
        postprocess_df=postprocess_df,
        do_save_to_latex=False,
    )

    for adapter, df in dfs.items():
        reshape_and_plot(df, output_folder, f"{basename}_{adapter}", name)
