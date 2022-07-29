# python validator_tests/process_df.py --exp_group_prefix mnist
# python validator_tests/process_df.py --exp_group_prefix office31
# python validator_tests/process_df.py --exp_group_prefix officehome

python simple_slurm.py --command "python validator_tests/process_df.py --detailed_warnings" --slurm_config_folder validator_tests --slurm_config mnist --job_name=process_df --cpus-per-task=2 --exp_group_excludes oracle