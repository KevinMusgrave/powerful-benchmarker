import pandas as pd

from powerful_benchmarker.utils import score_utils
from validator_tests.utils import df_utils


def shortened_task_name_dict():
    return {
        "mnist_mnist_mnistm": "MM",
        "office31_amazon_dslr": "AD",
        "office31_amazon_webcam": "AW",
        "office31_dslr_amazon": "DA",
        "office31_dslr_webcam": "DW",
        "office31_webcam_amazon": "WA",
        "office31_webcam_dslr": "WD",
        "officehome_art_clipart": "AC",
        "officehome_art_product": "AP",
        "officehome_art_real": "AR",
        "officehome_clipart_art": "CA",
        "officehome_clipart_product": "CP",
        "officehome_clipart_real": "CR",
        "officehome_product_art": "PA",
        "officehome_product_clipart": "PC",
        "officehome_product_real": "PR",
        "officehome_real_art": "RA",
        "officehome_real_clipart": "RC",
        "officehome_real_product": "RP",
    }


def shortened_task_names(df):
    return df.rename(columns=shortened_task_name_dict())


def convert_adapter_name(df):
    df["adapter"] = df["adapter"].str.replace("Config", "")


def add_source_only(df, accuracy_name):
    cols = df.columns.values
    _, split, average = df_utils.accuracy_name_split(accuracy_name)
    tasks_split = [df_utils.task_name_split(x) for x in cols]
    src_only_accs = [
        score_utils.pretrained_target_accuracy(
            dataset, [src_domains], [target_domains], split, average
        )
        for dataset, src_domains, target_domains in tasks_split
    ]
    src_only_accs = pd.DataFrame(
        {x: y for x, y in zip(cols, src_only_accs)}, index=["Source only"]
    )
    return pd.concat([src_only_accs, df], axis=0)


def resizebox(x):
    x = x.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}")
    x = x.replace("\\end{tabular}", "\\end{tabular}}")
    return x


# assuming unified validator/validator_args
def validators_to_remove():
    return [
        "Accuracy_average_macro_split_src_train",
        "Accuracy_average_macro_split_src_val",
        "Accuracy_average_macro_split_target_train",
        "Accuracy_average_macro_split_target_val",
        "Accuracy_average_micro_split_target_train",
        "Accuracy_average_micro_split_target_val",
        "BNM_layer_features_split_src_train",
        "BNM_layer_features_split_src_val",
        "BNM_layer_features_split_target_train",
        "BNM_layer_preds_split_src_train",
        "BNM_layer_preds_split_src_val",
        "BNM_layer_preds_split_target_train",
        "BNMSummed_layer_features",
        "BNMSummed_layer_preds",
        "BNMSummedSrcVal_layer_features",
        "BNMSummedSrcVal_layer_preds",
        "Diversity_split_src_train",
        "Diversity_split_src_val",
        "Diversity_split_target_train",
        "DiversitySummed",
    ]


def filter_validators(df):
    df = df_utils.unify_validator_columns(
        df, new_col_name="unified_validator", drop_validator_args=False
    )
    df = df.reset_index(drop=True)
    df = df.drop(df[df.unified_validator.isin(validators_to_remove())].index, axis=0)
    return df.drop(columns=["unified_validator"])


# Feed the output of validator_args_delimited into this
# def pretty_validator_args_dict(validator_args):
#     {"average_macro_split_src_train": "Src train, Macro",
#     "average_macro_split_src_val": "Src val, Macro",
#     "average_macro_split_target_train": "Target train, Macro",
#     "average_macro_split_target_val": "Target val, Macro",
#     "average_micro_split_src_train": "Src train, Micro"
#     "average_micro_split_src_val": "Src val, "}
