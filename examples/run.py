import logging
import argparse
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--pytorch_home", type=str, default=None)
parser.add_argument("--dataset_root", type=str, default="/home/datasets")
parser.add_argument("--root_experiment_folder", type=str, default="home/experiments")
parser.add_argument("--global_db_path", type=str, default=None)
parser.add_argument("--root_config_folder", type=str, default=None)
parser.add_argument("--bayes_opt_iters", type=int, default=0)
parser.add_argument("--reproductions", type=str, default="0")
args, _ = parser.parse_known_args()

if args.bayes_opt_iters > 0:
	from powerful_benchmarker.runners.bayes_opt_runner import BayesOptRunner
	args.reproductions = [int(x) for x in args.reproductions.split(",")]
	runner = BayesOptRunner
else:
	from powerful_benchmarker.runners.single_experiment_runner import SingleExperimentRunner
	runner = SingleExperimentRunner
	del args.bayes_opt_iters
	del args.reproductions

r = runner(**(args.__dict__))
r.run()