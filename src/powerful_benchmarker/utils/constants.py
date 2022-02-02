import yaml

BEST_TRIAL_FILENAME = "best_trial.json"


def get_user_constants():
    with open("constants.yaml", "r") as f:
        return yaml.safe_load(f)


def add_default_args(parser, args_to_add):
    constants = get_user_constants()
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
