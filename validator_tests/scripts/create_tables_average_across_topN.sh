prefixes=("mnist" "office31" "officehome")

for i in "${prefixes[@]}"
do
python validator_tests/create_tables_average_across_topN.py --exp_group_prefix $i --exp_group_excludes oracle --topN 1 10 100 1000
python validator_tests/create_tables_average_across_topN.py --exp_group_prefix $i --exp_group_excludes oracle --topN_per_adapter 1 10 100
done