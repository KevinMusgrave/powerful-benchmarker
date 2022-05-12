import argparse
import sys

sys.path.insert(0, ".")
from latex.best_accuracy_per_adapter import best_accuracy_per_adapter
from latex.correlation_src_threshold import correlation_src_threshold
from latex.predicted_best_acc import predicted_best_acc
from validator_tests.utils.constants import add_exp_group_args


def main(args):
    for topN in [1, 20]:
        best_accuracy_per_adapter(args, topN=topN)

    for threshold in [0, 0.9]:
        for per_adapter in [False, True]:
            correlation_src_threshold(
                args, threshold=threshold, per_adapter=per_adapter
            )
            topN_bounds = [1, 20] if per_adapter else [1, 200]
            for topN in topN_bounds:
                predicted_best_acc(
                    args, topN=topN, threshold=threshold, per_adapter=per_adapter
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_exp_group_args(parser)
    parser.add_argument("--input_folder", type=str, default="tables")
    parser.add_argument("--output_folder", type=str, default="tables_latex")
    args = parser.parse_args()
    main(args)
