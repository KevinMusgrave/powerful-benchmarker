import os

import pandas as pd

from latex.correlation import base_filename, get_postprocess_df, get_preprocess_df
from latex.table_creator import table_creator
from latex.utils import resizebox


def validator_parameter_explanations(args, name, src_threshold):
    basename = base_filename(name, False, src_threshold)

    df, output_folder = table_creator(
        args,
        args.input_folder,
        args.output_folder,
        basename,
        get_preprocess_df(False),
        get_postprocess_df(False),
        do_save_to_latex=False,
    )

    df = pd.DataFrame(index=df.index.copy()).reset_index()
    df = df.rename(columns={"level_0": "Validator", "level_1": "Parameters"})
    new_col = df.apply(explanations, axis=1)
    df["Explanation"] = new_col
    df = df.set_index(["Validator", "Parameters"])

    df_style = df.style
    latex_str = df_style.format(
        na_rep="-",
    ).to_latex(hrules=True, position_float="centering", clines="skip-last;data")

    latex_str = resizebox(latex_str)
    full_path = os.path.join(output_folder, "validator_parameter_explanations.tex")
    with open(full_path, "w") as text_file:
        text_file.write(latex_str)


def explanations(row):
    return {
        "Accuracy": {
            "Source Train": "\\texttt{Accuracy(Source train predictions)}",
            "Source Val": "\\texttt{Accuracy(Source validation predictions)}",
        },
        "BNM": {
            "Source Train": "\\texttt{BNM(Source train predictions)}",
            "Source Train + Target": "\\texttt{BNM(Source train predictions)} + \\texttt{BNM(Target predictions)}",
            "Source Val": "\\texttt{BNM(Source validation predictions)}",
            "Source Val + Target": "\\texttt{BNM(Source validation predictions)} + \\texttt{BNM(Target predictions)}",
            "Target": "\\texttt{BNM(Target predictions)}",
        },
        "ClassAMI": {
            "Source + Target Features": "\\texttt{ClassAMI(concat(Source train features, Target features))}",
            "Source + Target Logits": "\\texttt{ClassAMI(concat(Source train logits, Target logits))}",
            "Target Features": "\\texttt{ClassAMI(Target features)}",
            "Target Logits": "\\texttt{ClassAMI(Target logits)}",
        },
        "ClassSS": {
            "Source + Target Features": "\\texttt{ClassSS(concat(Source train normalized features, Target normalized features))}",
            "Source + Target Logits": "\\texttt{ClassSS(concat(Source train normalized logits, Target normalized logits))}",
            "Target Features": "\\texttt{ClassSS(Target normalized features)}",
            "Target Logits": "\\texttt{ClassSS(Target normalized logits)}",
        },
        "DEV": {
            "Features": "The discriminator is trained on feature vectors.",
            "Logits": "The discriminator is trained on logits.",
            "Preds": "The discriminator is trained on prediction vectors.",
        },
        "DEVN": {
            "Features, max normalization": "The discriminator is trained on feature vectors. The sample weights are max-normalized.",
            "Features, standardization": "The discriminator is trained on feature vectors. The sample weights are standardized.",
            "Logits, max normalization": "The discriminator is trained on logits. The sample weights are max-normalized.",
            "Logits, standardization": "The discriminator is trained on logits. The sample weights are standardized.",
            "Preds, max normalization": "The discriminator is trained on prediction vectors. The sample weights are max-normalized.",
            "Preds, standardization": "The discriminator is trained on prediction vectors. The sample weights are standardized.",
        },
        "Entropy": {
            "Source Train": "\\texttt{Entropy(Source train predictions)}",
            "Source Train + Target": "\\texttt{Entropy(Source train predictions)} + \\texttt{Entropy(Target predictions)}",
            "Source Val": "\\texttt{Entropy(Source validation predictions)}",
            "Source Val + Target": "\\texttt{Entropy(Source validation predictions)} + \\texttt{Entropy(Target predictions)}",
            "Target": "\\texttt{Entropy(Target predictions)}",
        },
        "SND": {
            "Features, $\\tau=0.05$": "The similarity matrix is derived from target features. Softmax temperature is 0.05.",
            "Features, $\\tau=0.1$": "The similarity matrix is derived from target features. Softmax temperature is 0.1.",
            "Features, $\\tau=0.5$": "The similarity matrix is derived from target features. Softmax temperature is 0.5.",
            "Logits, $\\tau=0.05$": "The similarity matrix is derived from target logits. Softmax temperature is 0.05.",
            "Logits, $\\tau=0.1$": "The similarity matrix is derived from target logits. Softmax temperature is 0.1",
            "Logits, $\\tau=0.5$": "The similarity matrix is derived from target logits. Softmax temperature is 0.5",
            "Preds, $\\tau=0.05$": "The similarity matrix is derived from target predictions. Softmax temperature is 0.05",
            "Preds, $\\tau=0.1$": "The similarity matrix is derived from target predictions. Softmax temperature is 0.1",
            "Preds, $\\tau=0.5$": "The similarity matrix is derived from target predictions. Softmax temperature is 0.5",
        },
    }[row.Validator][row.Parameters]
