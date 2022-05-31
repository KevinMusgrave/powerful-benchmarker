import yaml

TRIALS_FILENAME = "trials.csv"
BEST_TRIAL_FILENAME = "best_trial.json"
JOBIDS_FILENAME = "all_jobids.json"


def get_user_constants(constants_path):
    with open(constants_path, "r") as f:
        return yaml.safe_load(f)


def add_default_args(parser, args_to_add, constants_path="constants.yaml"):
    constants = get_user_constants(constants_path)
    for x in args_to_add:
        k, v = x, x
        if not isinstance(x, str):
            # then it's a sequence of strings
            k, v = x
        parser.add_argument(
            f"--{v}",
            type=str,
            default=constants[k],
        )
