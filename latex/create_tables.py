import argparse
import sys

sys.path.insert(0, ".")
from latex.best_accuracy_per_adapter import best_accuracy_per_adapter
from latex.correlation import correlation
from latex.correlation_bar_plot import correlation_bar_plot
from latex.correlation_diffs import correlation_diffs
from latex.correlation_single_adapter import correlation_single_adapter
from validator_tests.utils.constants import add_exp_group_args


def main(args):
    correlation_names = ["weighted_spearman", "spearman"]
    best_accuracy_per_adapter(args)
    correlation_diffs(args, False, correlation_names)

    for per_adapter in [False, True]:
        for name in correlation_names:
            correlation(args, per_adapter, name)
            correlation_bar_plot(args, per_adapter, name)
            if per_adapter:
                correlation_single_adapter(args, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_exp_group_args(parser)
    parser.add_argument("--input_folder", type=str, default="tables")
    parser.add_argument("--output_folder", type=str, default="tables_latex")
    parser.add_argument("--nlargest", type=int, default=5)
    args = parser.parse_args()
    main(args)
