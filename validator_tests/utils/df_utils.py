from .utils import dict_to_str, validator_str

SPLIT_NAMES = ["src_train", "src_val", "target_train", "target_val"]
AVERAGE_NAMES = ["micro", "macro"]


def exp_specific_columns(df):
    exclude = ["score", "validator", "validator_args", *all_acc_score_column_names()]
    return [x for x in df.columns.values if x not in exclude]


def acc_score_column_name(split, average):
    return f"{split}_{average}"


def all_acc_score_column_names():
    return [acc_score_column_name(x, y) for x in SPLIT_NAMES for y in AVERAGE_NAMES]


def get_acc_rows(df, split, average):
    args = dict_to_str({"average": average, "split": split})
    return df[(df["validator_args"] == args) & (df["validator"] == "Accuracy")]


def get_acc_df(df, split, average):
    df = get_acc_rows(df, split, average)
    df = df.drop(columns=["validator", "validator_args"])
    return df.rename(columns={"score": acc_score_column_name(split, average)})


def get_all_acc(df):
    output = None
    for split in SPLIT_NAMES:
        for average in ["micro", "macro"]:
            curr = get_acc_df(df, split, average)
            if output is None:
                output = curr
            else:
                output = output.merge(curr, on=exp_specific_columns(output))
    return output


# need to do this to avoid pd hash error
def convert_list_to_tuple(df):
    df.src_domains = df.src_domains.apply(tuple)
    df.target_domains = df.target_domains.apply(tuple)


def assert_acc_rows_are_correct(df):
    # make sure score and split/average columns are equal
    for split in SPLIT_NAMES:
        for average in ["micro", "macro"]:
            curr = get_acc_rows(df, split, average)
            if not curr["score"].equals(curr[acc_score_column_name(split, average)]):
                raise ValueError("These columns should be equal")


def domains_str(domains):
    return "_".join(domains)


def task_str(dataset, src_domains, target_domains):
    return f"{dataset}_{domains_str(src_domains)}_{domains_str(target_domains)}"


def add_task_column(df):
    return df.assign(
        task=lambda x: task_str(x["dataset"], x["src_domains"], x["target_domains"])
    )


def unify_validator_columns(df):
    new_col = df.apply(
        lambda x: validator_str(x["validator"], x["validator_args"]), axis=1
    )
    df = df.assign(validator=new_col)
    return df.drop(columns=["validator_args"])


def maybe_per_adapter(df, per_adapter):
    if per_adapter:
        adapters = df["adapter"].unique()
    else:
        adapters = [None]
    return adapters
