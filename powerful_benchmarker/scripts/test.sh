# python -m powerful_benchmarker.main --exp_name test_experiment0 --dataset mnist \
# --src_domains mnist --adapter PretrainerConfig \
# --download_datasets --num_trials 2 \
# --max_epochs 20 --pretrain_on_src --validator src_accuracy \
# --use_stat_getter

# python -m powerful_benchmarker.main --exp_name test_experiment0 --target_domains mnist mnistm --evaluate --validator oracle

# python -m powerful_benchmarker.main --exp_name test_experiment1 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets --start_with_pretrained \
# --feature_layer 6 --max_epochs 3 --num_trials 2

# python -m powerful_benchmarker.main --exp_name test_experiment2 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets --start_with_pretrained \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --validator oracle \
# --num_reproduce 2

# python -m powerful_benchmarker.main --exp_name test_experiment3 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets --start_with_pretrained \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --validator oracle --use_stat_getter

# python -m powerful_benchmarker.main --exp_name test_experiment4 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets --start_with_pretrained \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --use_stat_getter

# python -m powerful_benchmarker.main --exp_name test_experiment5 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets --start_with_pretrained \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --save_features



python -m powerful_benchmarker.main --exp_name dann_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter DANNConfig \
--download_datasets --start_with_pretrained \
--feature_layer 6 --max_epochs 2 --num_trials 1 \
--save_features --use_full_inference

python -m powerful_benchmarker.main --exp_name mcc_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter MCCConfig \
--download_datasets --start_with_pretrained \
--feature_layer 6 --max_epochs 2 --num_trials 1 \
--save_features --use_full_inference

python -m powerful_benchmarker.main --exp_name atdoc_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter ATDOCConfig \
--download_datasets --start_with_pretrained \
--feature_layer 6 --max_epochs 2 --num_trials 1 \
--save_features --use_full_inference

python -m powerful_benchmarker.main --exp_name bsp_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter BSPConfig \
--download_datasets --start_with_pretrained \
--feature_layer 6 --max_epochs 2 --num_trials 1 \
--save_features --use_full_inference

python -m powerful_benchmarker.main --exp_name cdan_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter CDANConfig \
--download_datasets --start_with_pretrained \
--feature_layer 6 --max_epochs 2 --num_trials 1 \
--save_features --use_full_inference

python -m powerful_benchmarker.main --exp_name im_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter IMConfig \
--download_datasets --start_with_pretrained \
--feature_layer 6 --max_epochs 2 --num_trials 1 \
--save_features --use_full_inference

python -m powerful_benchmarker.main --exp_name bnm_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter BNMConfig \
--download_datasets --start_with_pretrained \
--feature_layer 6 --max_epochs 2 --num_trials 1 \
--save_features --use_full_inference


python -m powerful_benchmarker.main --exp_name gvb_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter GVBConfig \
--download_datasets --start_with_pretrained \
--feature_layer 6 --max_epochs 2 --num_trials 1 \
--save_features --use_full_inference

python -m powerful_benchmarker.main --exp_name mcd_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter MCDConfig \
--download_datasets --start_with_pretrained \
--feature_layer 6 --max_epochs 2 --num_trials 1 \
--save_features --use_full_inference

python -m powerful_benchmarker.main --exp_name mmd_test --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter MMDConfig \
--download_datasets --start_with_pretrained \
--feature_layer 6 --max_epochs 2 --num_trials 1 \
--save_features --use_full_inference