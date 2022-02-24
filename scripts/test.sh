# python main.py --exp_name test_experiment0 --dataset mnist \
# --src_domains mnist --adapter PretrainerConfig \
# --download_datasets --num_trials 2 \
# --max_epochs 20 --pretrain_on_src --validator src_accuracy \
# --use_stat_getter

# python main.py --exp_name test_experiment0 --target_domains mnist mnistm --evaluate --validator oracle

# python main.py --exp_name test_experiment1 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets --start_with_pretrained \
# --feature_layer 6 --max_epochs 3 --num_trials 2

# python main.py --exp_name test_experiment2 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets --start_with_pretrained \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --validator oracle \
# --num_reproduce 2

# python main.py --exp_name test_experiment3 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets --start_with_pretrained \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --validator oracle --use_stat_getter

# python main.py --exp_name test_experiment4 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets --start_with_pretrained \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --use_stat_getter

# python main.py --exp_name test_experiment5 --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets --start_with_pretrained \
# --feature_layer 6 --max_epochs 3 --num_trials 2 --save_features



python main.py --exp_name test_experiment2 --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter DANNConfig \
--download_datasets --start_with_pretrained \
--feature_layer 6 --max_epochs 3 --num_trials 2 --validator oracle \
--num_reproduce 2