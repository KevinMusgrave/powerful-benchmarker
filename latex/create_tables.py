import argparse
import sys

sys.path.insert(0, ".")
from latex.best_accuracy_per_adapter import best_accuracy_per_adapter
from latex.correlation import correlation
from latex.correlation_bar_plot import correlation_bar_plot
from latex.correlation_bar_plot_adapter_validator_pairs import (
    correlation_bar_plot_adapter_validator_pairs,
)
from latex.correlation_bar_plot_single_adapter import (
    correlation_bar_plot_single_adapter,
)
from latex.correlation_diffs import correlation_diffs
from latex.correlation_single_adapter import correlation_single_adapter
from latex.pred_acc_using_best_adapter_validator_pairs import (
    pred_acc_using_best_adapter_validator_pairs,
)
from latex.validator_parameter_explanations import validator_parameter_explanations
from validator_tests.utils.constants import add_exp_group_args


def main(args):
    src_threshold = 0.0
    wsp = "weighted_spearman"
    best_accuracy_per_adapter(args)
    validator_parameter_explanations(args, wsp, src_threshold)
    pred_acc_using_best_adapter_validator_pairs(args, wsp, src_threshold)
    correlation_diffs(args, False, [wsp, "spearman"], src_threshold)
    correlation_single_adapter(args, wsp, src_threshold)
    correlation_bar_plot_single_adapter(args, wsp, src_threshold)
    correlation_bar_plot_adapter_validator_pairs(args, wsp, src_threshold)

    for per_adapter in [False, True]:
        correlation(args, per_adapter, wsp, src_threshold)
        correlation_bar_plot(args, per_adapter, wsp, src_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_exp_group_args(parser)
    add_exp_group_args(parser, "_select_best")
    parser.add_argument("--input_folder", type=str, default="tables")
    parser.add_argument("--output_folder", type=str, default="tables_latex")
    parser.add_argument("--nlargest", type=int, default=5)
    args = parser.parse_args()
    main(args)
