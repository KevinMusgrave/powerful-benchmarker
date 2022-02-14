# python main.py --exp_name test_experiment --dataset mnist \
# --src_domains mnist --target_domains mnistm --adapter DANNConfig \
# --download_datasets --start_with_pretrained --save_features \
# --feature_layer 6 --max_epochs 3

# python main.py --exp_name test_experiment --dataset mnist \
# --src_domains mnist --adapter PretrainerConfig \
# --download_datasets --save_features --num_trials 3 \
# --max_epochs 3 --pretrain_on_src --validator src_accuracy

python main.py --exp_name test_experiment --target_domains mnist mnistm --evaluate --validator oracle