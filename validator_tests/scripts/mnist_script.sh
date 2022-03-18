# python -m scripts.run_validators --exp_group mnist_mnist_mnistm_fl6_Adam_lr1 --exp_names atdoc bnm bsp cdan dann gvb im mcc mmd --flags Accuracy --slurm_config mnist --exp_per_slurm_job 4 --trials_per_exp 100 --run

# python -m scripts.run_validators --exp_group mnist_mnist_mnistm_fl6_Adam_lr1 --exp_names atdoc bnm bsp cdan dann gvb im mcc mmd --flags Entropy --slurm_config mnist --exp_per_slurm_job 4 --trials_per_exp 100 --run

# python -m scripts.run_validators --exp_group mnist_mnist_mnistm_fl6_Adam_lr1 --exp_names atdoc bnm bsp cdan dann gvb im mcc mmd --flags Diversity --slurm_config mnist --exp_per_slurm_job 4 --trials_per_exp 100 --run

# python -m scripts.run_validators --exp_group mnist_mnist_mnistm_fl6_Adam_lr1 --exp_names atdoc bnm bsp cdan dann gvb im mcc mmd --flags DEV --slurm_config mnist --exp_per_slurm_job 4 --trials_per_exp 100 --run --time=8:00:00

# python -m scripts.run_validators --exp_group mnist_mnist_mnistm_fl6_Adam_lr1 --exp_names atdoc bnm bsp cdan dann gvb im mcc mmd --flags SND --slurm_config mnist --exp_per_slurm_job 3 --trials_per_exp 100 --run --time=8:00:00

python -m scripts.run_validators --exp_group mnist_mnist_mnistm_fl6_Adam_lr1 --exp_names atdoc bnm bsp cdan dann gvb im mcc mmd --flags KNN --slurm_config mnist --exp_per_slurm_job 4 --trials_per_exp 100 --run --time=8:00:00

python -m scripts.run_validators --exp_group mnist_mnist_mnistm_fl6_Adam_lr1 --exp_names atdoc bnm bsp cdan dann gvb im mcc mmd --flags IST --slurm_config mnist --exp_per_slurm_job 2 --trials_per_exp 100 --run --time=8:00:00

