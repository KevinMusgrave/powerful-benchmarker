exp_prefixes=("mnist_mnist_mnistm" "office31_amazon_dslr" "office31_amazon_webcam" "office31_dslr_amazon" \
"office31_dslr_webcam" "office31_webcam_amazon" "office31_webcam_dslr" \
"officehome_art_clipart" "officehome_art_product" "officehome_art_real" \
"officehome_clipart_art" "officehome_clipart_product" "officehome_clipart_real" \
"officehome_product_art" "officehome_product_clipart" "officehome_product_real" \
"officehome_real_art" "officehome_real_clipart" "officehome_real_product")


for i in "${exp_prefixes[@]}"
do
    python simple_slurm.py --command "python validator_tests/save_resilience_to_noise_dfs.py --run_combined" --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests \
    --slurm_config mnist --job_name=resilience_to_noise --cpus-per-task=8 --exp_group_excludes oracle \
    --gres=gpu:0
done


exp_prefixes=("domainnet126_clipart_painting" "domainnet126_clipart_real" "domainnet126_clipart_sketch" \
"domainnet126_painting_clipart" "domainnet126_painting_real" "domainnet126_painting_sketch" \
"domainnet126_real_clipart" "domainnet126_real_painting" "domainnet126_real_sketch" \
"domainnet126_sketch_clipart" "domainnet126_sketch_painting" "domainnet126_sketch_real")


for i in "${exp_prefixes[@]}"
do
    python simple_slurm.py --command "python validator_tests/save_resilience_to_noise_dfs.py --run_single" --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests \
    --slurm_config mnist --job_name=resilience_to_noise --cpus-per-task=8 --exp_group_excludes oracle \
    --gres=gpu:0
done

