import argparse
import sys

sys.path.insert(0, ".")
from powerful_benchmarker.utils.constants import add_default_args
from validator_tests.utils import create_main
from validator_tests.utils.constants import add_exp_group_args
from validator_tests.utils.plot_score_vs_epoch import plot_score_vs_epoch
from validator_tests.utils.plot_val_vs_acc import plot_val_vs_acc


def scatter(plots_folder, df, per_feature_layer):
    kwargs = {}
    if args.no_color:
        kwargs["c"] = None
    if args.dot_size:
        kwargs["s"] = args.dot_size
    if args.font_scale:
        kwargs["font_scale"] = args.font_scale
    if args.figsize:
        kwargs["figsize"] = args.figsize
    plot_val_vs_acc(
        df,
        plots_folder,
        per_adapter=args.per_adapter,
        per_feature_layer=per_feature_layer,
        validator_set=args.validator_set,
        src_threshold=args.src_threshold,
        adapter=args.adapter,
        **kwargs
    )


def score_vs_epoch(plots_folder, df, per_feature_layer):
    plot_score_vs_epoch(
        df,
        plots_folder,
        per_adapter=args.per_adapter,
        per_feature_layer=per_feature_layer,
        validator_set=args.validator_set,
        src_threshold=args.src_threshold,
        adapter=args.adapter,
    )


def get_fns(fn_list):
    if fn_list == []:
        fn_list = ["scatter", "over_time"]

    def fn1(*args):
        if "scatter" in fn_list:
            scatter(*args, per_feature_layer=True)
        if "score_vs_epoch" in fn_list:
            score_vs_epoch(*args, per_feature_layer=True)

    def fn2(*args):
        if "scatter" in fn_list:
            scatter(*args, per_feature_layer=False)
        if "score_vs_epoch" in fn_list:
            score_vs_epoch(*args, per_feature_layer=False)

    return fn1, fn2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    add_exp_group_args(parser)
    create_main.add_main_args(parser)
    parser.add_argument("--output_folder", type=str, default="plots")
    parser.add_argument("--validator_set", nargs="+", type=str, default=None)
    parser.add_argument("--src_threshold", type=float)
    parser.add_argument("--no_color", action="store_true")
    parser.add_argument("--dot_size", type=float, default=None)
    parser.add_argument("--font_scale", type=float, default=None)
    parser.add_argument("--figsize", nargs="+", type=float, default=None)
    parser.add_argument("--per_adapter", action="store_true")
    parser.add_argument("--adapter", type=str)
    parser.add_argument("--fn_list", nargs="+", type=str, default=[])
    args = parser.parse_args()
    create_main.main(args, *get_fns(args.fn_list))
