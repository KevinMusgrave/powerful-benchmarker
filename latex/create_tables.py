import argparse
import sys

sys.path.insert(0, ".")
from latex.best_accuracy_per_adapter import best_accuracy_per_adapter
from latex.correlation_src_threshold import correlation_src_threshold
from validator_tests.utils.constants import add_exp_group_args


def main(args):
    best_accuracy_per_adapter(args)
    correlation_src_threshold(args, threshold=0)
    correlation_src_threshold(args, threshold=0.9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_exp_group_args(parser)
    parser.add_argument("--input_folder", type=str, default="tables")
    parser.add_argument("--output_folder", type=str, default="tables_latex")
    args = parser.parse_args()
    main(args)
