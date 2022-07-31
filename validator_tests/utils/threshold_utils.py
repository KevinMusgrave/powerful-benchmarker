def filter_by_acc(df, min_acc, domain_type):
    if domain_type == "src":
        split = "val"
    elif domain_type == "target":
        split = "train"
    return df[df[f"{domain_type}_{split}_micro"] >= min_acc]
