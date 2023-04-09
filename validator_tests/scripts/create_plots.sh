exp_prefixes=("mnist_mnist_mnistm" "office31_amazon_dslr" "office31_amazon_webcam" "office31_dslr_amazon" \
"office31_dslr_webcam" "office31_webcam_amazon" "office31_webcam_dslr" \
"officehome_art_clipart" "officehome_art_product" "officehome_art_real" \
"officehome_clipart_art" "officehome_clipart_product" "officehome_clipart_real" \
"officehome_product_art" "officehome_product_clipart" "officehome_product_real" \
"officehome_real_art" "officehome_real_clipart" "officehome_real_product")

domainnet_exp_prefixes=("domainnet126_clipart_painting" "domainnet126_clipart_real" "domainnet126_clipart_sketch" \
"domainnet126_painting_clipart" "domainnet126_painting_real" "domainnet126_painting_sketch" \
"domainnet126_real_clipart" "domainnet126_real_painting" "domainnet126_real_sketch" \
"domainnet126_sketch_clipart" "domainnet126_sketch_painting" "domainnet126_sketch_real")

# for i in "${exp_prefixes[@]}"
# do
#     python simple_slurm.py --command "python validator_tests/create_plots.py --no_color --validator_set BNMSummedSrcVal --adapter ATDOCConfig --run_combined --per_adapter" --all_in_one --exp_group_prefix $i --exp_group_excludes oracle \
#     --slurm_config_folder validator_tests --slurm_config mnist --job_name=eval_validators --cpus-per-task=4
# done


# Bigger plots with small dots
# python validator_tests/create_plots.py --exp_group_prefix officehome_art_real --no_color --validator_set SND --run_combined --figsize 10 10 --font_scale 2


# Plots with less data should use bigger dots
# python validator_tests/create_plots.py --exp_group_prefix office31_dslr_webcam --no_color --validator_set BNMSummedSrcVal --adapter ATDOCConfig --per_adapter --run_combined --figsize 10 10 --font_scale 2 --dot_size 1.5



python validator_tests/create_plots.py --no_color --validator_set ClassSSCentroidInit --run_combined --exp_group_prefix officehome_art_real --figsize 10 10 --font_scale 2
python validator_tests/create_plots.py --no_color --validator_set Entropy --run_combined --exp_group_prefix officehome_real_art --figsize 10 10 --font_scale 2
python validator_tests/create_plots.py --no_color --validator_set DEVBinary --run_combined --exp_group_prefix officehome_clipart_art --figsize 10 10 --font_scale 2
python validator_tests/create_plots.py --no_color --validator_set ClassAMICentroidInit ClassSSCentroidInit --run_combined --exp_group_prefix officehome_real_clipart --figsize 10 10 --font_scale 2
python validator_tests/create_plots.py --no_color --validator_set SND --run_combined --exp_group_prefix officehome_clipart_real --figsize 10 10 --font_scale 2 --dot_size 1 --adapter CDANConfig --per_adapter
python validator_tests/create_plots.py --no_color --validator_set SND --run_combined --exp_group_prefix officehome_product_clipart --figsize 10 10 --font_scale 2 --dot_size 1 --adapter MMDConfig --per_adapter
python validator_tests/create_plots.py --no_color --validator_set ClassSSCentroidInit --run_combined --exp_group_prefix mnist --figsize 10 10 --font_scale 2


# for i in "${exp_prefixes[@]}"
# do
# python simple_slurm.py --command "python validator_tests/create_plots.py --no_color --validator_set ClassAMICentroidInit --run_combined --figsize 10 10 --font_scale 2 --dot_size 1 --adapter MCCConfig --per_adapter" \
# --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests --slurm_config mnist --job_name=create_plots --cpus-per-task=4 --gres=gpu:0
# python simple_slurm.py --command "python validator_tests/create_plots.py --no_color --validator_set ClassAMICentroidInit --run_combined --figsize 10 10 --font_scale 2 --dot_size 1 --adapter IMConfig --per_adapter" \
# --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests --slurm_config mnist --job_name=create_plots --cpus-per-task=4 --gres=gpu:0
# python simple_slurm.py --command "python validator_tests/create_plots.py --no_color --validator_set ClassAMICentroidInit --run_combined --figsize 10 10 --font_scale 2 --dot_size 1 --adapter BNMConfig --per_adapter" \
# --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests --slurm_config mnist --job_name=create_plots --cpus-per-task=4 --gres=gpu:0
# python simple_slurm.py --command "python validator_tests/create_plots.py --no_color --validator_set BNM --run_combined --figsize 10 10 --font_scale 2 --dot_size 1 --adapter ATDOCConfig --per_adapter" \
# --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests --slurm_config mnist --job_name=create_plots --cpus-per-task=4 --gres=gpu:0
# python simple_slurm.py --command "python validator_tests/create_plots.py --no_color --validator_set Accuracy --run_combined --figsize 10 10 --font_scale 2 --dot_size 1 --per_adapter" \
# --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests --slurm_config mnist --job_name=create_plots --cpus-per-task=4 --gres=gpu:0
# done


# for i in "${domainnet_exp_prefixes[@]}"
# do
# python simple_slurm.py --command "python validator_tests/create_plots.py --no_color --validator_set ClassAMICentroidInit --run_single --figsize 10 10 --font_scale 2 --dot_size 1 --adapter MCCConfig --per_adapter" \
# --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests --slurm_config mnist --job_name=create_plots --cpus-per-task=4 --gres=gpu:0
# python simple_slurm.py --command "python validator_tests/create_plots.py --no_color --validator_set ClassAMICentroidInit --run_single --figsize 10 10 --font_scale 2 --dot_size 1 --adapter IMConfig --per_adapter" \
# --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests --slurm_config mnist --job_name=create_plots --cpus-per-task=4 --gres=gpu:0
# python simple_slurm.py --command "python validator_tests/create_plots.py --no_color --validator_set ClassAMICentroidInit --run_single --figsize 10 10 --font_scale 2 --dot_size 1 --adapter BNMConfig --per_adapter" \
# --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests --slurm_config mnist --job_name=create_plots --cpus-per-task=4 --gres=gpu:0
# python simple_slurm.py --command "python validator_tests/create_plots.py --no_color --validator_set BNM --run_single --figsize 10 10 --font_scale 2 --dot_size 1 --adapter ATDOCConfig --per_adapter" \
# --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests --slurm_config mnist --job_name=create_plots --cpus-per-task=4 --gres=gpu:0
# python simple_slurm.py --command "python validator_tests/create_plots.py --no_color --validator_set Accuracy --run_single --figsize 10 10 --font_scale 2 --dot_size 1 --per_adapter" \
# --all_in_one --exp_group_prefix $i --slurm_config_folder validator_tests --slurm_config mnist --job_name=create_plots --cpus-per-task=4 --gres=gpu:0
# done


# python validator_tests/create_plots.py --no_color --validator_set ClassAMICentroidInit --exp_group_prefix domainnet --run_single --figsize 10 10 --font_scale 2 --dot_size 1 --adapter MCCConfig --per_adapter
# python validator_tests/create_plots.py --no_color --validator_set ClassAMICentroidInit --exp_group_prefix domainnet --run_single --figsize 10 10 --font_scale 2 --dot_size 1 --adapter IMConfig --per_adapter
# python validator_tests/create_plots.py --no_color --validator_set ClassAMICentroidInit --exp_group_prefix domainnet --run_single --figsize 10 10 --font_scale 2 --dot_size 1 --adapter BNMConfig --per_adapter
# python validator_tests/create_plots.py --no_color --validator_set BNM --exp_group_prefix domainnet --run_single --figsize 10 10 --font_scale 2 --dot_size 1 --adapter ATDOCConfig --per_adapter
# python validator_tests/create_plots.py --no_color --validator_set Accuracy--exp_group_prefix domainnet --run_single --figsize 10 10 --font_scale 2 --dot_size 1 --per_adapter