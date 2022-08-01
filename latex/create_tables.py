import argparse
import sys

sys.path.insert(0, ".")
from latex.best_accuracy_per_adapter import best_accuracy_per_adapter
from latex.correlation_src_threshold import correlation_src_threshold
from latex.correlation_src_threshold_single_adapter import (
    correlation_src_threshold_single_adapter,
)
from latex.weighted_spearman import weighted_spearman
from latex.weighted_spearman_single_adapter import weighted_spearman_single_adapter
from validator_tests.utils.constants import add_exp_group_args


def main(args):
    # best_accuracy_per_adapter(args, topN=topN)

    for per_adapter in [False, True]:
        # correlation_src_threshold(
        #     args, threshold=threshold, per_adapter=per_adapter
        # )
        # correlation_src_threshold_single_adapter(args, threshold=threshold)
        weighted_spearman(args, per_adapter=per_adapter)
        if per_adapter:
            weighted_spearman_single_adapter(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_exp_group_args(parser)
    parser.add_argument("--input_folder", type=str, default="tables")
    parser.add_argument("--output_folder", type=str, default="tables_latex")
    args = parser.parse_args()
    main(args)
