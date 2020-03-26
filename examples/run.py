import logging
import argparse
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--pytorch_home", type=str, default="/home/tkm45/NEW_STUFF/pytorch_models")
parser.add_argument("--dataset_root", type=str, default="/scratch")
parser.add_argument("--root_experiment_folder", type=str, default="/home/tkm45/NEW_STUFF/experiments")
parser.add_argument("--global_db_path", type=str, default=None)
parser.add_argument("--root_config_folder", type=str, default=None)
parser.add_argument("--bayes_opt_iters", type=int, default=0)
parser.add_argument("--num_reproductions", type=int, default=0)
args, _ = parser.parse_known_args()

if args.bayes_opt_iters > 0:
	from powerful_benchmarker.runners.bayes_opt_runner import BayesOptRunner
	runner = BayesOptRunner
else:
	from powerful_benchmarker.runners.single_experiment_runner import SingleExperimentRunner
	runner = SingleExperimentRunner
	del args.bayes_opt_iters
	del args.num_reproductions

r = runner(**(args.__dict__))
r.run()
