python main.py --exp_name test_experiment --dataset mnist \
--src_domains mnist --target_domains mnistm --adapter DANNConfig \
--download_datasets --start_with_pretrained --save_features \
--feature_layer 6 --max_epochs 3