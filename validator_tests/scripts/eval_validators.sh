exp_prefixes=("mnist_mnist_mnistm" "office31_amazon_dslr" "office31_amazon_webcam" "office31_dslr_amazon" \
"office31_dslr_webcam" "office31_webcam_amazon" "office31_webcam_dslr" \
"officehome_art_clipart" "officehome_art_product" "officehome_art_real" \
"officehome_clipart_art" "officehome_clipart_product" "officehome_clipart_real" \
"officehome_product_art" "officehome_product_clipart" "officehome_product_real" \
"officehome_real_art" "officehome_real_clipart" "officehome_real_product")


for i in "${exp_prefixes[@]}"
do
    python simple_slurm.py --command "python validator_tests/eval_validators.py --run_combined" --all_in_one --exp_group_prefix $i --exp_group_excludes oracle \
    --slurm_config_folder validator_tests --slurm_config mnist --job_name=per_src_threshold --cpus-per-task=4
done