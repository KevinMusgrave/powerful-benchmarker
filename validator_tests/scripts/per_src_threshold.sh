python validator_tests/per_src_threshold.py --run_single --run_combined --exp_group_prefix mnist
python validator_tests/per_src_threshold.py --run_single --run_combined --exp_group_prefix mnist --topN 1 --topN_per_adapter 1
python validator_tests/per_src_threshold.py --run_single --run_combined --exp_group_prefix office31
python validator_tests/per_src_threshold.py --run_single --run_combined --exp_group_prefix office31 --topN 1 --topN_per_adapter 1
python validator_tests/per_src_threshold.py --run_single --exp_group_prefix officehome
python validator_tests/per_src_threshold.py --run_single --exp_group_prefix officehome --topN 1 --topN_per_adapter 1
python validator_tests/per_src_threshold.py --run_combined --exp_group_prefix officehome
python validator_tests/per_src_threshold.py --run_combined --exp_group_prefix officehome --topN 1 --topN_per_adapter 1
