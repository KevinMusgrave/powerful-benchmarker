prefixes=("mnist" "office31" "officehome")

for i in "${prefixes[@]}"
do
python validator_tests/create_tables.py --exp_group_prefix $i --exp_group_excludes oracle --run_combined --topN 1 --topN_per_adapter 1
python validator_tests/create_tables.py --exp_group_prefix $i --exp_group_excludes oracle --run_combined --topN 10 --topN_per_adapter 10
python validator_tests/create_tables.py --exp_group_prefix $i --exp_group_excludes oracle --run_combined --topN 100 --topN_per_adapter 100
python validator_tests/create_tables.py --exp_group_prefix $i --exp_group_excludes oracle --run_combined --topN 1000
done