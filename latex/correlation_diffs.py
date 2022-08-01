import os

from latex.correlation import get_postprocess_df, get_preprocess_df
from latex.table_creator import table_creator


def correlation_diffs(args, per_adapter, names):
    dfs = []
    for name in names:
        per_adapter_str = "_per_adapter" if per_adapter else ""
        basename = f"{name}{per_adapter_str}"

        df, output_folder = table_creator(
            args,
            basename,
            preprocess_df=get_preprocess_df(per_adapter),
            postprocess_df=get_postprocess_df(per_adapter),
            do_save_to_latex=False,
        )

        df = (
            df.drop(columns=["Mean", "Std"])
            .reset_index()
            .melt(id_vars=["level_0", "level_1"], var_name="task", value_name=name)
        )

        dfs.append(df)

    df = dfs[0].merge(dfs[1], on=["level_0", "level_1", "task"])
    df["diff"] = df["weighted_spearman"] - df["spearman"]
    df = df.sort_values("diff")
    df.to_csv(os.path.join(output_folder, "correlation_diffs.csv"))
