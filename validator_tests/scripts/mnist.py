import subprocess

components = [
    {"flags": "Accuracy", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "Entropy", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "Diversity", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "DEV", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
    {"flags": "SND", "exp_per_slurm_job": "3", "trials_per_exp": "100"},
    {"flags": "KNN", "exp_per_slurm_job": "4", "trials_per_exp": "100"},
]

exp_names = ["atdoc", "bnm", "bsp", "cdan", "dann", "gvb", "im", "mcc", "mmd"]
exp_names = " ".join(exp_names)

for x in components:
    command = "python validator_tests/run_validators.py --exp_groups mnist_mnist_mnistm_fl6_Adam_lr1 --slurm_config mnist --run --time=8:00:00"
    command += f" --exp_names {exp_names} --flags {x['flags']} --exp_per_slurm_job {x['exp_per_slurm_job']} --trials_per_exp {x['trials_per_exp']}"
    subprocess.run(command.split(" "))
