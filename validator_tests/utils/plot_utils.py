import os

import tqdm

from .utils import domains_str, maybe_per_adapter, validator_args_str


def create_name(filters, components):
    s = ""
    for k in components:
        curr_val = filters[k]
        if k in ["src_domains", "target_domains"]:
            x = domains_str(curr_val)
        elif k == "validator_args":
            x = validator_args_str(curr_val)
        else:
            x = curr_val
        s = x if s == "" else f"{s}_{x}"
    return s


def filter_df(df, filter_by, kwargs, finished_list):
    finished = {k: kwargs[k] for k in filter_by}
    if finished in finished_list:
        return []
    finished_list.append(finished)
    for k in filter_by:
        df = df[df[k] == kwargs[k]]
    return df


def plot_loop(
    df,
    plots_folder,
    plot_fn,
    filter_by,
    sub_folder_components,
    filename_components,
    filename=None,
    per_adapter=True,
):
    finished = []
    if not per_adapter and "adapter" in filter_by:
        raise ValueError("Can't do per_adapter=False and filter by adapter")
    adapters = maybe_per_adapter(df, per_adapter)
    for adapter in adapters:
        for dataset in df["dataset"].unique():
            for src_domains in df["src_domains"].unique():
                for target_domains in df["target_domains"].unique():
                    for validator in df["validator"].unique():
                        print(
                            "plotting", dataset, src_domains, target_domains, validator
                        )
                        for args in tqdm.tqdm(df["validator_args"].unique()):
                            filters = {
                                "adapter": adapter,
                                "dataset": dataset,
                                "src_domains": src_domains,
                                "target_domains": target_domains,
                                "validator": validator,
                                "validator_args": args,
                            }
                            curr_df = filter_df(
                                df,
                                filter_by,
                                filters,
                                finished,
                            )
                            if len(curr_df) == 0:
                                continue
                            curr_plots_folder = os.path.join(
                                plots_folder,
                                create_name(filters, sub_folder_components),
                            )
                            if len(filename_components) > 0:
                                filename = create_name(filters, filename_components)
                            plot_fn(curr_plots_folder, curr_df, filename)
