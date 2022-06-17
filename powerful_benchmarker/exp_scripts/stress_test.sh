python powerful_benchmarker/launch_multiple.py --exp_config test/stress_test --slurm_config a100 --partition=hipri --gpus-per-node=2
python powerful_benchmarker/launch_multiple.py --exp_config test/stress_test --slurm_config a100 --partition=lowpri --gpus-per-node=2
python powerful_benchmarker/launch_multiple.py --exp_config test/stress_test --slurm_config a100 --partition=lowpri --gpus-per-node=2