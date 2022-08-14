python powerful_benchmarker/main.py --exp_name test_experiment0 --dataset mnist \
--src_domains mnist --adapter PretrainerConfig \
--download_datasets --num_trials 2 \
--max_epochs 2 --pretrain_on_src --validator src_accuracy \
--use_stat_getter

python powerful_benchmarker/main.py --exp_name test_experiment0 --target_domains mnist mnistm --evaluate --validator oracle

# python powerful_benchmarker/main.py --exp_name test_experiment1 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 3 --num_trials 2

# python powerful_benchmarker/main.py --exp_name test_experiment2 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --validator oracle \
# --num_reproduce 2

# python powerful_benchmarker/main.py --exp_name test_experiment3 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --validator oracle --use_stat_getter

# python powerful_benchmarker/main.py --exp_name test_experiment4 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --use_stat_getter

# python powerful_benchmarker/main.py --exp_name test_experiment5 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --save_features



# python powerful_benchmarker/main.py --exp_name dann_test --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 2 --num_trials 1 \
# --save_features --use_full_inference

# python powerful_benchmarker/main.py --exp_name mcc_test --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter MCCConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 2 --num_trials 1 \
# --save_features --use_full_inference

# python powerful_benchmarker/main.py --exp_name atdoc_test --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter ATDOCConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 2 --num_trials 1 \
# --save_features --use_full_inference

# python powerful_benchmarker/main.py --exp_name bsp_test --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter BSPConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 2 --num_trials 1 \
# --save_features --use_full_inference

# python powerful_benchmarker/main.py --exp_name cdan_test --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter CDANConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 2 --num_trials 1 \
# --save_features --use_full_inference

# python powerful_benchmarker/main.py --exp_name im_test --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter IMConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 2 --num_trials 1 \
# --save_features --use_full_inference

# python powerful_benchmarker/main.py --exp_name bnm_test --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter BNMConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 2 --num_trials 1 \
# --save_features --use_full_inference


# python powerful_benchmarker/main.py --exp_name gvb_test --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter GVBConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 2 --num_trials 1 \
# --save_features --use_full_inference

# python powerful_benchmarker/main.py --exp_name mcd_test --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter MCDConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 2 --num_trials 1 \
# --save_features --use_full_inference

# python powerful_benchmarker/main.py --exp_name mmd_test --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter MMDConfig \
# --download_datasets \
# --feature_layer 6 --max_epochs 2 --num_trials 1 \
# --save_features --use_full_inference

# for domain in "clipart" "painting" "real" "sketch"
# do

#     exp_name=pretrained_domainnet126_${domain}

#     python powerful_benchmarker/main.py \
#     --exp_name ${exp_name} --dataset domainnet126 \
#     --src_domains ${domain} --adapter PretrainerConfig \
#     --num_trials 5 --batch_size 32 --num_workers 2 --n_startup_trials 5 \
#     --max_epochs 100 --patience 10 --pretrain_on_src --validator src_accuracy \
#     --optimizer SGD --pretrain_lr 0.01 --check_initial_score

#     for validator in "oracle" "oracle_micro"
#     do
#         python powerful_benchmarker/main.py --exp_name ${exp_name} \
#         --target_domains clipart painting real sketch --evaluate --validator ${validator}
#     done

# done