# A Metric Learning Reality Check

This page contains additional information for the [ECCV 2020 paper](https://arxiv.org/abs/2003.08505) by Musgrave et al.

## Optimization plots

Click on the links below to view the bayesian optimization plots. 

These are also available in the [benchmark spreadsheet](https://docs.google.com/spreadsheets/d/1brUBishNxmld-KLDAJewIc43A4EVZk3gY6yKe8OIKbY/).

| CUB200 | Cars196 | SOP | CUB200 with Batch 256 |
|-|-|-|-|
| [Contrastive](mlrc_plots/cub_contrastive.html){target=_blank} | [Contrastive](mlrc_plots/cars_contrastive.html){target=_blank} | [Contrastive](mlrc_plots/sop_contrastive.html){target=_blank} | [Contrastive](mlrc_plots/cub_contrastive_large_batch.html){target=_blank} |
| [Triplet](mlrc_plots/cub_triplet.html){target=_blank} | [Triplet](mlrc_plots/cars_triplet.html){target=_blank} | [Triplet](mlrc_plots/sop_triplet.html){target=_blank} | [Triplet](mlrc_plots/cub_triplet_large_batch.html){target=_blank} |
| [NTXent](mlrc_plots/cub_ntxent.html){target=_blank} | [NTXent](mlrc_plots/cars_ntxent.html){target=_blank} | [NTXent](mlrc_plots/sop_ntxent.html){target=_blank} | [NTXent](mlrc_plots/cub_ntxent_large_batch.html){target=_blank} |
| [ProxyNCA](mlrc_plots/cub_proxy_nca.html){target=_blank} | [ProxyNCA](mlrc_plots/cars_proxy_nca.html){target=_blank} | [ProxyNCA](mlrc_plots/sop_proxy_nca.html){target=_blank} | [ProxyNCA](mlrc_plots/cub_proxy_nca_large_batch.html){target=_blank} |
| [Margin](mlrc_plots/cub_margin_no_weight_decay.html){target=_blank} | [Margin](mlrc_plots/cars_margin_no_weight_decay.html){target=_blank} | [Margin](mlrc_plots/sop_margin_no_weight_decay.html){target=_blank} | [Margin](mlrc_plots/cub_margin_large_batch_no_weight_decay.html){target=_blank} |
| [Margin / class](mlrc_plots/cub_margin_param_per_class_no_weight_decay.html){target=_blank} | [Margin / class](mlrc_plots/cars_margin_param_per_class_no_weight_decay.html){target=_blank} | [Margin / class](mlrc_plots/sop_margin_param_per_class_no_weight_decay.html){target=_blank} | [Margin / class](mlrc_plots/cub_margin_param_per_class_large_batch_no_weight_decay.html){target=_blank} |
| [Normalized Softmax](mlrc_plots/cub_normalized_softmax.html){target=_blank} | [Normalized Softmax](mlrc_plots/cars_normalized_softmax.html){target=_blank} | [Normalized Softmax](mlrc_plots/sop_normalized_softmax.html){target=_blank} | [Normalized Softmax](mlrc_plots/cub_normalized_softmax_large_batch.html){target=_blank} |
| [CosFace](mlrc_plots/cub_cosface.html){target=_blank} | [CosFace](mlrc_plots/cars_cosface.html){target=_blank} | [CosFace](mlrc_plots/sop_cosface.html){target=_blank} | [CosFace](mlrc_plots/cub_cosface_large_batch.html){target=_blank} |
| [ArcFace](mlrc_plots/cub_arcface.html){target=_blank} | [ArcFace](mlrc_plots/cars_arcface.html){target=_blank} | [ArcFace](mlrc_plots/sop_arcface.html){target=_blank} | [ArcFace](mlrc_plots/cub_arcface_large_batch.html){target=_blank} |
| [FastAP](mlrc_plots/cub_fast_ap.png){target=_blank} | [FastAP](mlrc_plots/cars_fast_ap.png){target=_blank} | [FastAP](mlrc_plots/sop_fast_ap.png){target=_blank} | [FastAP](mlrc_plots/cub_fastap_large_batch.html){target=_blank} |
| [SNR Contrastive](mlrc_plots/cub_snr_contrastive.html){target=_blank} | [SNR Contrastive](mlrc_plots/cars_snr_contrastive.html){target=_blank} | [SNR Contrastive](mlrc_plots/sop_snr_contrastive.html){target=_blank} | [SNR Contrastive](mlrc_plots/cub_snr_contrastive_large_batch.html){target=_blank} |
| [Multi Similarity](mlrc_plots/cub_multi_similarity.html){target=_blank} | [Multi Similarity](mlrc_plots/cars_multi_similarity.html){target=_blank} | [Multi Similarity](mlrc_plots/sop_multi_similarity.html){target=_blank} | [Multi Similarity](mlrc_plots/cub_multi_similarity_large_batch.html){target=_blank} |
| [Multi Similarity + Miner](mlrc_plots/cub_multi_similarity_with_ms_miner.html){target=_blank} | [Multi Similarity + Miner](mlrc_plots/cars_multi_similarity_with_ms_miner.html){target=_blank} | [Multi Similarity + Miner](mlrc_plots/sop_multi_similarity_with_ms_miner.html){target=_blank} | [Multi Similarity + Miner](mlrc_plots/cub_multi_similarity_with_ms_miner_large_batch_wider_range.html){target=_blank} |
| [SoftTriple](mlrc_plots/cub_soft_triple.html){target=_blank} | [SoftTriple](mlrc_plots/cars_soft_triple.html){target=_blank} | [SoftTriple](mlrc_plots/sop_soft_triple.html){target=_blank} | [SoftTriple](mlrc_plots/cub_soft_triple_large_batch_wider_range.html){target=_blank} |


## Optimal hyperparameters

The values below are also available in the [benchmark spreadsheet](https://docs.google.com/spreadsheets/d/1brUBishNxmld-KLDAJewIc43A4EVZk3gY6yKe8OIKbY/).

Loss function | CUB200 | Cars196 | SOP | CUB200 with Batch 256 |
|-|-|-|-|-|
|**Contrastive**<br/>pos_margin<br/>neg_margin|<br/>-0.2000<br/>0.3841|<br/>0.2652<br/>0.5409|<br/>0.2850<br/>0.5130|<br/>0.2227<br/>0.7694|
|**Triplet**<br/>margin|<br/>0.0961|<br/>0.1190|<br/>0.0451|<br/>0.1368|
|**NTXent**<br/>temperature|<br/>0.0091|<br/>0.0219|<br/>0.0002|<br/>0.0415|
|**ProxyNCA**<br/>proxy lr<br/>softmax_scale|<br/>6.04e-3<br/>13.98|<br/>4.43e-3<br/>7.97|<br/>5.28e-4<br/>10.73|<br/>2.16e-1<br/>10.03|
|**Margin**<br/>beta lr<br/>margin<br/>init beta|<br/>1.31e-3<br/>0.0878<br/>0.7838<br/>|<br/>1.11e-4<br/>0.0781<br/>1.3164<br/>|<br/>1.82e-3<br/>0.0915<br/>1.1072|<br/>1.00e-6<br/>0.0674<br/>0.9762|
|**Margin / class**<br/>beta lr<br/>margin<br/>init beta|<br/>2.65e-4<br/>0.0779<br/>0.9796|<br/>4.76e-05<br/>0.0776<br/>0.9598|<br/>7.10e-05<br/>0.0518<br/>0.8424|<br/>1.32e-2<br/>-0.0204<br/>0.1097|
|**Normalized Softmax**<br/>weights lr<br/>temperature|<br/>4.46e-3<br/>0.1087|<br/>1.10e-2<br/>0.0886|<br/>5.46e-4<br/>0.0630|<br/>7.20e-2<br/>0.0707|
|**CosFace**<br/>weights lr<br/>margin<br/>scale<br/>|<br/>2.53e-3<br/>0.6182<br/>100.0|<br/>7.41e-3<br/>0.4324<br/>161.5|<br/>2.16e-3<br/>0.3364<br/>100.0|<br/>3.99e-3<br/>0.4144<br/>88.23|
|**ArcFace**<br/>weights lr<br/>margin<br/>scale<br/>|<br/>5.13e-3<br/>23.22<br/>100.0|<br/>7.39e-06<br/>20.52<br/>49.50|<br/>2.01e-3<br/>18.63<br/>220.3|<br/>3.95e-2<br/>23.14<br/>78.86|
|**FastAP**<br/>num_bins|<br/>17|<br/>27|<br/>16|<br/>86|
|**SNR Contrastive**<br/>pos_margin<br/>neg_margin<br/>regularizer_weight|<br/>0.3264<br/>0.8446<br/>0.1382|<br/>0.1670<br/>0.9337<br/>0|<br/>0.3759<br/>1.0831<br/>0|<br/>0.1182<br/>0.6822<br/>0.4744|
|**Multi Similarity**<br/>alpha<br/>beta<br/>base|<br/>0.01<br/>50.60<br/>0.56|<br/>14.35<br/>75.83<br/>0.66|<br/>8.49<br/>57.38<br/>0.41|<br/>0.01<br/>46.85<br/>0.82|
|**Multi Similarity + Miner**<br/>alpha<br/>beta<br/>base<br/>epsilon|<br/>17.97<br/>75.66<br/>0.77<br/>0.39|<br/>7.49<br/>47.99<br/>0.63<br/>0.72|<br/>15.94<br/>156.61<br/>0.72<br/>0.34|<br/>11.63<br/>55.20<br/>0.85<br/>0.42|
|**SoftTriple**<br/>weights lr<br/>la<br/>gamma<br/>reg_weight<br/>margin|<br/>5.37e-05<br/>78.02<br/>58.95<br/>0.3754<br/>0.4307|<br/>1.40e-4<br/>17.69<br/>19.18<br/>0.0669<br/>0.3588|<br/>8.68e-05<br/>100.00<br/>47.90<br/>N/A<br/>0.3145|<br/>1.06e-4<br/>72.12<br/>51.07<br/>0.4430<br/>0.6959|

## Frequently Asked Questions

#### Do you have slides that accompany the paper?
Slides are [here](https://docs.google.com/presentation/d/1KnLDFzMKLYlnMzMDc7wyKHVAh5dJ9z6Fs1qto4OqQFY/edit?usp=sharing){target=_blank}.

#### Do you have examples to back up the claims in section 2.1 of the paper?
See [this document](https://docs.google.com/document/d/1xx56SwR2a0JMWaiHgi2oIdCaSlMFMPMPlh_RBM94paw/edit?usp=sharing){target=_blank}.

#### Isn't it unfair to fix the model, optimizer, learning rate, and embedding size?
Our goal was to compare algorithms fairly. To accomplish this, we used the same network, optimizer, learning rate, image transforms, and embedding dimensionality for each algorithm. There is no theoretical reason why changing any of these parameters would benefit one particular algorithm over the rest. If there is no theoretical reason, then we can only speculate, and if we add hyperparameters based on speculation, then the search space becomes too large to explore.

#### Why was the batch size set to 32 for most of the results?
This was done for the sake of computational efficiency. Note that there are: 

- 3 datasets 
- 14 algorithms 
- 50 steps of bayesian optmization 
- 4 fold cross validation 

This comes to 8400 models to train, which can take a considerable amount of time. Thus, a batch size of 32 made sense. It's also important to remember that there are real-world cases where a large batch size cannot be used. For example, if you want to train on large images, rather than the contrived case of 227x227, then training with a batch size of 32 suddenly makes a lot more sense because you are constrained by GPU memory. So it's reasonable to check the performance of these losses on a batch size of 32. 

That said, there is a good theoretical reason for a larger batch size benefiting embedding losses more than classification losses. Specifically, embedding losses can benefit from the increased number of pairs/triplets in larger batches. To address this, we benchmarked the 14 methods on CUB200, using a batch size of 256. The results can be found in the supplementary section (the final page) of the paper.


#### Why weren't more hard-mining methods evaluated?
We did test one loss+miner combination (Multi-similarity loss + their mining method). But we mainly wanted to do a thorough evaluation of loss functions, because that is the subject of most recent metric learning papers.   


#### For the contrastive loss, why is the optimal positive margin a negative value?

A negative value should be equivalent to a margin of 0, because the distance between positive pairs cannot be negative, and the margin does not contribute to the gradient. So allowing the hyperparameter optimization to explore negative margins was unnecesary, but by the time I realized this, it wasn't worth changing the optimization bounds.


## Reproducing results
### Download the experiment folder

1. Go to the [benchmark spreadsheet](https://docs.google.com/spreadsheets/d/1brUBishNxmld-KLDAJewIc43A4EVZk3gY6yKe8OIKbY/)
2. Find the experiment you want to reproduce, and click on the link in the "Config files" column.
3. You'll see 3 folders: one for CUB, one for Cars, and one for SOP. Open the folder for the dataset you want to train on.
4. Now you'll see several files and folders, one of which ends in "reproduction0". Download this folder. (It will include saved models. If you don't want to download the saved models, go into the folder and download just the "configs" folder.)

### Command line scripts
Normally reproducing results is as easy as downloading an experiment folder, and [using the ```reproduce_results``` flag](../index.md#reproduce-an-experiment). However, there have been significant changes to the API since these experiments were run, so there are a couple of extra steps required, and they depend on the dataset. 

Additionally, if you are reproducing an experiment for the **Contrastive, Triplet, or SNR Contrastive losses**, you have to delete the key/value pair called ```avg_non_zero_only``` in the ```config_loss_and_miners.yaml``` file. And for the **Contrastive loss**, you should delete the ```use_similarity``` key/value pair in ```config_loss_and_miners.yaml```. 



In the following code, ```<experiment_to_reproduce>``` refers to the folder that **contains** the ```configs``` folder.

  - CUB200:

```bash
python run.py --reproduce_results <experiment_to_reproduce> \
--experiment_name <your_experiment_name> \
--split_manager~SWAP~1 {MLRCSplitManager: {}} \
--merge_argparse_when_resuming
```

  - Cars196:

```bash
python run.py --reproduce_results <experiment_to_reproduce> \
--experiment_name <your_experiment_name> \
--config_dataset [default, with_cars196] \
--config_general [default, with_cars196] \
--split_manager~SWAP~1 {MLRCSplitManager: {}} \
--merge_argparse_when_resuming
```

  - Stanford Online Products

```bash
python run.py --reproduce_results <experiment_to_reproduce> \
--experiment_name <your_experiment_name> \
--config_dataset [default, with_sop] \
--config_general [default, with_sop] \
--split_manager~SWAP~1 {MLRCSplitManager: {}} \
--merge_argparse_when_resuming
```

  - CUB200 with batch size 256:
```bash
python run.py --reproduce_results <experiment_to_reproduce> \
--experiment_name <your_experiment_name> \
--config_general [default, with_256_batch] \
--split_manager~SWAP~1 {MLRCSplitManager: {}} \
--merge_argparse_when_resuming
```


If you don't have the datasets and would like to download them into your ```dataset_root``` folder, you can add this flag to the CUB commands:
```bash
--dataset~OVERRIDE~ {CUB200: {download: True}}
```

Likewise, for the Cars196 and Stanford Online Products commands, replace the ```--config_dataset``` flag with:
```bash
--dataset~OVERRIDE~ {Cars196: {download: True}}
```
or 
```bash
--dataset~OVERRIDE~ {StanfordOnlineProducts: {download: True}}
```

### Run evaluation on the test set
After training is done, you can get the "separate 128-dim" test set performance:
```bash
python run.py --experiment_name <your_experiment_name> \
--evaluate --splits_to_eval [test]
```
and the "concatenated 512-dim" test set performance:
```bash
python run.py --experiment_name <your_experiment_name> \
--evaluate_ensemble --splits_to_eval [test]
```

Once evaluation is done, you can go to the ```meta_logs``` folder and view the results.
