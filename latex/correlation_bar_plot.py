import os

import seaborn as sns

from latex.correlation import get_postprocess_df, get_preprocess_df
from latex.table_creator import table_creator


def correlation_bar_plot(args, per_adapter, name):
    per_adapter_str = "_per_adapter" if per_adapter else ""
    basename = f"{name}{per_adapter_str}"

    df, output_folder = table_creator(
        args,
        basename,
        preprocess_df=get_preprocess_df(per_adapter),
        postprocess_df=get_postprocess_df(per_adapter),
        do_save_to_latex=False,
    )

    df = df.reset_index()
    new_col = df.apply(lambda x: f'{x["level_0"]}: {x["level_1"]}', axis=1)
    df = df.assign(validator_as_str=new_col).drop(columns=["level_0", "level_1"])
    dfm = df.drop(columns=["Mean", "Std"]).melt(id_vars=["validator_as_str"])
    assert "Mean" not in dfm["variable"].values
    assert "Std" not in dfm["variable"].values
    order = df.sort_values("Mean", ascending=False)["validator_as_str"].values
    xlabel = (
        "Weighed Spearman Correlation"
        if name == "weighted_spearman"
        else "Spearman Correlation"
    )

    sns.set(style="whitegrid", rc={"figure.figsize": (12, 12)})
    plt = sns.barplot(x="value", y="validator_as_str", data=dfm, ci="sd", order=order)
    plt.set(xlabel=xlabel, ylabel="Validator")
    fig = plt.get_figure()
    fig.savefig(
        os.path.join(output_folder, f"{basename}_barplot.png"), bbox_inches="tight"
    )
    fig.clf()
