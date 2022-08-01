from powerful_benchmarker.utils.score_utils import pretrained_src_accuracy

from . import df_utils


def filter_by_acc(df, min_acc, domain_type, filter_action):
    if domain_type == "src":
        split = "val"
    elif domain_type == "target":
        split = "train"

    keep_mask = df[f"{domain_type}_{split}_micro"] >= min_acc
    if filter_action == "remove":
        return df[keep_mask]
    elif filter_action == "set_to_nan":
        df = df.copy()
        df.loc[~keep_mask, "score"] = float("nan")
        return df
    else:
        raise ValueError("incorrect filter_action")


def filter_by_src_threshold(df, src_threshold, filter_action):
    dataset = df_utils.get_sorted_unique(df, "dataset", assert_one=True)[0]
    src_domains = df_utils.get_sorted_unique(df, "src_domains", assert_one=True)[0]
    min_acc = pretrained_src_accuracy(dataset, src_domains, "val", "micro")
    return filter_by_acc(df, min_acc * src_threshold, "src", filter_action)
