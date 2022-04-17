import argparse
import subprocess


def main(args):
    validators = [
        {"flags": "Accuracy", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
        {"flags": "Entropy", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
        {"flags": "Diversity", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
        {"flags": "DEV", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
        {"flags": "DEVBinary", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
        {"flags": "SND", "exp_per_slurm_job": "3", "trials_per_exp": "100"},
        {"flags": "DomainCluster", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
        {
            "flags": "ClassAMICentroidInit",
            "exp_per_slurm_job": "4",
            "trials_per_exp": "100",
        },
        {"flags": "MMD", "exp_per_slurm_job": "4", "trials_per_exp": "50"},
        {"flags": "MMDPerClass", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
        {"flags": "MMDFixedB", "exp_per_slurm_job": "4", "trials_per_exp": "50"},
        {
            "flags": "MMDPerClassFixedB",
            "exp_per_slurm_job": "4",
            "trials_per_exp": "100",
        },
        {"flags": "TargetKNN", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
        {"flags": "TargetKNNLogits", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
        {"flags": "BSP", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
        {"flags": "BNM", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    ]

    validators = [v for v in validators if v["flags"] in args.validators]

    exp_names = [
        "atdoc",
        "bnm",
        "bsp",
        "cdan",
        "dann",
        "gvb",
        "im",
        "mcc",
        "mcd",
        "mmd",
    ]
    if args.exp_names is not None:
        exp_names = args.exp_names
    exp_names = " ".join(exp_names)

    for x in validators:
        command = f"python validator_tests/run_validators.py {args.other_args} --slurm_config {args.slurm_config} --run"
        command += f" --exp_names {exp_names} --flags {x['flags']} --exp_per_slurm_job {x['exp_per_slurm_job']} --trials_per_exp {x['trials_per_exp']}"
        subprocess.run(command.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--exp_names", nargs="+", type=str, default=None)
    parser.add_argument("--validators", nargs="+", type=str, default=[])
    parser.add_argument("--other_args", type=str, default="")
    parser.add_argument("--slurm_config", type=str, required=True)
    args = parser.parse_args()
    main(args)
