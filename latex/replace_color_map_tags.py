import argparse
import glob
import os
import sys

sys.path.insert(0, ".")
from latex.utils import num_to_word


def main(args):
    matches = glob.glob(
        os.path.join(
            args.input_folder, "**", "best_accuracy_per_adapter_ranked_by_score*"
        )
    )
    for m in matches:
        nlargest = os.path.splitext(m)[0].split("_")[-1]
        with open(m, "r") as f:
            num = num_to_word(int(nlargest))
            newText = f.read().replace(
                f"bestaccuracyperadapterrankedbyscore{num}",
                f"bestaccuracyperadapter{num}",
            )

        with open(m, "w") as f:
            f.write(newText)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input_folder", type=str, default="tables_latex")
    args = parser.parse_args()
    main(args)
