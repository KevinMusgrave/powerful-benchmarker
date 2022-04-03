import subprocess

components = [
    {"flags": "Accuracy", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "Entropy", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "Diversity", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "DEV", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "DEVBinary", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "SND", "exp_per_slurm_job": "3", "trials_per_exp": "100"},
    {"flags": "KNN", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "AMI", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "MMD", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "MMDPerClass", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "MMDFixedB", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "MMDPerClassFixedB", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "DLogitsAccuracy", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "TargetKNN", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "TargetKNNLogits", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "BSP", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "BNM", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "SilhouetteScore", "exp_per_slurm_job": "6", "trials_per_exp": "50"},
    {"flags": "CHScore", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
]

all_exps = ["atdoc", "bnm", "bsp", "cdan", "dann", "gvb", "im", "mcc", "mcd", "mmd"]
exp_with_d = ["cdan", "dann", "gvb"]


for x in components:
    if x["flags"] in ["DLogitsAccuracy"]:
        exp_names = " ".join(exp_with_d)
    else:
        exp_names = " ".join(all_exps)
    command = "python validator_tests/run_validators.py --exp_groups mnist_mnist_mnistm_fl6_Adam_lr1 --slurm_config mnist --run --time=8:00:00"
    command += f" --exp_names {exp_names} --flags {x['flags']} --exp_per_slurm_job {x['exp_per_slurm_job']} --trials_per_exp {x['trials_per_exp']}"
    subprocess.run(command.split(" "))
