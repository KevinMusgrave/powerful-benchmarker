# A Metric Learning Reality Check

This page contains additional information for the [ECCV 2020 paper](https://arxiv.org/abs/2003.08505) by Musgrave et al.

## Optimization plots

Click on the links below to view the bayesian optimization plots. These are also available in the [benchmark spreadsheet](https://docs.google.com/spreadsheets/d/1brUBishNxmld-KLDAJewIc43A4EVZk3gY6yKe8OIKbY/).

The plots were generated using the [Ax](https://github.com/facebook/Ax){target=_blank} package.



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
|**Contrastive**<br/>pos_margin<br/>neg_margin|<br/>-0.2000<sup>[1](../mlrc/#for-the-contrastive-loss-why-is-the-optimal-positive-margin-a-negative-value)</sup><br/>0.3841|<br/>0.2652<br/>0.5409|<br/>0.2850<br/>0.5130|<br/>0.2227<br/>0.7694|
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

## Examples of unfair comparisons in metric learning papers

#### Papers that use a better architecture than their competitors, but don’t disclose it

  - [Sampling Matters in Deep Embedding Learning (ICCV 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf)

    - Uses ResNet50, but all competitors use GoogleNet

  - [Deep Metric Learning with Hierarchical Triplet Loss (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ge_Deep_Metric_Learning_ECCV_2018_paper.pdf)

    - Uses BN-Inception, but all competitors use GoogleNet

  - [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

    - Uses BN-Inception. Claims better performance than ensemble methods, but the ensemble methods use GoogleNet.

  - [Deep Metric Learning to Rank (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf)

    - Uses ResNet50. In their SOP table, only 1 out of 11 competitor methods use ResNet50. All others use BN-Inception or GoogleNet. Claims better performance than ensemble methods, but the ensemble methods use GoogleNet. 

  - [Divide and Conquer the Embedding Space for Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf)

    - Uses ResNet50. In their Cars196 and SOP tables, only 1 out of 15 competitor methods use ResNet50. The rest use GoogleNet or BN-Inception. The same is true for their CUB200 results, but in that table, they re-implement two of the competitors to use ResNet50.

  - [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf)
  
    - Uses BN-Inception. Compares with N-pairs and HDC, but doesn’t mention that these use GoogleNet. They only mention the competitors’ architectures when the competitors use an equal or superior network. Specifically, they mention that the Margin loss uses ResNet50, and HTL uses BN-Inception.

  - [Deep Metric Learning with Tuplet Margin Loss (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf)
  
    - Uses ResNet50. In their SOP table, only 1 out of 10 competitors use ResNet50, and in their CUB200 and Cars196 tables, only 1 out of 8 competitors use ResNet50. The rest use GoogleNet or BN-Inception. They also claim better performance than ensemble methods, but the ensemble methods use GoogleNet.



#### Papers that use a higher dimensionality than their competitors, but don’t disclose it

  - [Sampling Matters in Deep Embedding Learning (ICCV 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf)

    - Uses size 128. CUB200 table: 4 out of 7 use size 64. Cars196: 4 out of 5 use size 64. SOP: 4 out of 7 use size 64.

  - [Deep Metric Learning with Hierarchical Triplet Loss (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ge_Deep_Metric_Learning_ECCV_2018_paper.pdf)

    - Uses size 512. The top two non-ensemble competitor results use size 384 and 64.

  - [Ranked List Loss for Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
    
    - Uses size 512 or 1536. For all 3 datasets, 5 out of the 6 competitor results use size 64.

  - [Deep Metric Learning with Tuplet Margin Loss (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf)
    
    - Uses size 512. The only competing method that uses the same architecture, uses size 128.


#### Papers that claim to do a simple 256 resize and 227 or 224 random crop, but actually use the more advanced RandomResizedCrop method

  - [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

    - [Link to line in code](https://github.com/MalongTech/research-ms-loss/blob/master/ret_benchmark/data/transforms/build.py#L17){target=_blank}

  - [Divide and Conquer the Embedding Space for Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf)

    - [Link to line in code](https://github.com/CompVis/metric-learning-divide-and-conquer/blob/master/lib/data/set/transform.py#L51){target=_blank}

  - [MIC: Mining Interclass Characteristics for Improved Metric Learning (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.pdf)

    - [Link to line in code](https://github.com/Confusezius/ICCV2019_MIC/blob/master/datasets.py#L324){target=_blank}

  - [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf)

    - [Link to line in code](https://github.com/idstcv/SoftTriple/blob/master/train.py#L99){target=_blank}

  - [Proxy Anchor Loss for Deep Metric Learning (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)

    - [Link to line in code](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/master/code/dataset/utils.py#L76){target=_blank}


#### Papers that use a 256 crop size, but whose competitor results use a smaller 227 or 224 size

  - [Metric Learning With HORDE: High-Order Regularizer for Deep Embeddings (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jacob_Metric_Learning_With_HORDE_High-Order_Regularizer_for_Deep_Embeddings_ICCV_2019_paper.pdf)

    - Although they do reimplement some algorithms, and the reimplementations presumably use a crop size of 256, they also compare to paper results that use 227 or 224.

#### Papers that omit details

  - [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

    - [Freezes batchnorm parameters in their code](https://github.com/MalongTech/research-ms-loss/blob/master/ret_benchmark/utils/freeze_bn.py){target=_blank}, but this is not mentioned in the paper.
  
  - [Proxy Anchor Loss for Deep Metric Learning (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)
    
    - Uses the [sum of Global Average Pooling (GAP) and Global Max Pooling (GMP)](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/issues/1){target=_blank}. Competitor papers use just GAP. This is not mentioned in the paper. 


## Examples to back up other claims in section 2.1

#### “Most papers claim to apply the following transformations: resize the image to 256 x 256, randomly crop to 227 x 227, and do a horizontal flip with 50% chance”. The following papers support this claim

 - [Deep Metric Learning via Lifted Structured Feature Embedding (CVPR 2016)](https://arxiv.org/pdf/1511.06452.pdf)
 - [Deep Spectral Clustering Learning (ICML 2017)](https://www.cs.toronto.edu/~urtasun/publications/law_etal_icml17.pdf)
 - [Deep Metric Learning via Facility Location (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Song_Deep_Metric_Learning_CVPR_2017_paper.pdf)
 - [No Fuss Distance Metric Learning using Proxies (ICCV 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.pdf)
 - [Deep Metric Learning with Angular Loss (ICCV 2017)](https://arxiv.org/pdf/1708.01682.pdf)
 - [Sampling Matters in Deep Embedding Learning (ICCV 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf)
 - [Deep Adversarial Metric Learning (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf)
 - [Classification is a Strong Baseline for Deep Metric Learning (BMVC 2019)](https://labs.pinterest.com/user/themes/pin_labs/assets/paper/classification-strong-baseline-bmvc-2019.pdf)
 - [Hardness-Aware Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Hardness-Aware_Deep_Metric_Learning_CVPR_2019_paper.pdf)
 - [Deep Asymmetric Metric Learning via Rich Relationship Mining (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Deep_Asymmetric_Metric_Learning_via_Rich_Relationship_Mining_CVPR_2019_paper.pdf)
 - [Stochastic Class-based Hard Example Mining for Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Suh_Stochastic_Class-Based_Hard_Example_Mining_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
 - [Ranked List Loss for Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
 - [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
 - [Deep Metric Learning to Rank (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf)
 - [Divide and Conquer the Embedding Space for Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf)
 - [MIC: Mining Interclass Characteristics for Improved Metric Learning (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.pdf)
 - [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf)
 - [Proxy Anchor Loss for Deep Metric Learning (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)


#### Papers categorized by the optimizer they use

 - SGD:
 
	- [Deep Spectral Clustering Learning (ICML 2017)](https://www.cs.toronto.edu/~urtasun/publications/law_etal_icml17.pdf)
	- [Deep Metric Learning with Angular Loss (ICCV 2017)](https://arxiv.org/pdf/1708.01682.pdf)
	- [Hard-Aware Deeply Cascaded Embedding (ICCV 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf)
	- [Deep Metric Learning with Hierarchical Triplet Loss (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ge_Deep_Metric_Learning_ECCV_2018_paper.pdf)
	- [Deep Asymmetric Metric Learning via Rich Relationship Mining (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Deep_Asymmetric_Metric_Learning_via_Rich_Relationship_Mining_CVPR_2019_paper.pdf)
	- [Ranked List Loss for Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
	- [Classification is a Strong Baseline for Deep Metric Learning (BMVC 2019)](https://labs.pinterest.com/user/themes/pin_labs/assets/paper/classification-strong-baseline-bmvc-2019.pdf)
	- [Deep Metric Learning with Tuplet Margin Loss (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf)

 - RMSprop:

	- [Deep Metric Learning via Facility Location (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Song_Deep_Metric_Learning_CVPR_2017_paper.pdf)
	- [No Fuss Distance Metric Learning using Proxies (ICCV 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.pdf)

 - Adam:

	- [Improved Deep Metric Learning with Multi-class N-pair Loss Objective (Neurips 2016)](https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf)
	- [Sampling Matters in Deep Embedding Learning (ICCV 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf)
	- [Hybrid-Attention based Decoupled Metric Learning for Zero-Shot Image Retrieval (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Hybrid-Attention_Based_Decoupled_Metric_Learning_for_Zero-Shot_Image_Retrieval_CVPR_2019_paper.pdf)
	- [Stochastic Class-based Hard Example Mining for Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Suh_Stochastic_Class-Based_Hard_Example_Mining_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
	- [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
	- [Deep Metric Learning to Rank (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf)
	- [Divide and Conquer the Embedding Space for Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf)
	- [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf)
	- [Metric Learning With HORDE: High-Order Regularizer for Deep Embeddings (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jacob_Metric_Learning_With_HORDE_High-Order_Regularizer_for_Deep_Embeddings_ICCV_2019_paper.pdf)
	- [MIC: Mining Interclass Characteristics for Improved Metric Learning (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.pdf)

 - AdamW

	- [Proxy Anchor Loss for Deep Metric Learning (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)


#### Papers that do not use confidence intervals
 - All of the previously mentioned papers



#### Papers that do not use a validation set
 - All of the previously mentioned papers


## What papers report for the contrastive and triplet losses

The tables below are what papers have reported for the contrastive and triplet loss, **using convnets**. We know that the papers are reporting convnet results because they explicitly say so. For example:

* [Lifted Structure Loss](https://arxiv.org/pdf/1511.06452.pdf): See figures 6, 7, and 12, which indicate that the contrastive and triplet results were obtained using GoogleNet. These results have been cited several times in recent papers.
* [Deep Adversarial Metric Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf): See tables 1, 2, and 3, and this quote from the bottom of page 6 / top of page 7: "For all the baseline methods and DAML, we employed the same GoogLeNet architecture pre-trained on ImageNet for fair comparisons"
* [Hardness-Aware Deep Metric Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Hardness-Aware_Deep_Metric_Learning_CVPR_2019_paper.pdf): See tables 1, 2, and 3, and this quote from page 8: "We evaluated all the methods mentioned above using the same pretrained CNN model for fair comparison."

#### Reported Precision@1 for the Contrastive Loss
| Paper | CUB200 | Cars196 | SOP |
|-|-|-|-|
|[Deep Metric Learning via Lifted Structured Feature Embedding (CVPR 2016)](https://arxiv.org/pdf/1511.06452.pdf)|26.4|21.7|42|
|[Learning Deep Embeddings with Histogram Loss (NIPS 2016)](https://arxiv.org/pdf/1611.00822.pdf)|26.4|N/A|42|
|[Hard-Aware Deeply Cascaded Embedding (ICCV 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf)|26.4|21.7|42|
|[Sampling Matters in Deep Embedding Learning (ICCV 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf)|N/A|N/A|30.1|
|[Deep Adversarial Metric Learning (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf)|27.2|27.6|37.5|
|[Attention-based Ensemble for Deep Metric Learning (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Wonsik_Kim_Attention-based_Ensemble_for_ECCV_2018_paper.pdf)|26.4|21.7|42|
|[Deep Variational Metric Learning (ECCV 2018)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Hardness-Aware_Deep_Metric_Learning_CVPR_2019_paper.pdf)|32.8|35.8|37.4|
|[Classification is a Strong Baseline for Deep Metric Learning (BMVC 2019)](https://labs.pinterest.com/user/themes/pin_labs/assets/paper/classification-strong-baseline-bmvc-2019.pdf)|26.4|21.7|42|
|[Deep Asymmetric Metric Learning via Rich Relationship Mining (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Deep_Asymmetric_Metric_Learning_via_Rich_Relationship_Mining_CVPR_2019_paper.pdf)|27.2|27.6|37.5|
|[Hardness-Aware Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Hardness-Aware_Deep_Metric_Learning_CVPR_2019_paper.pdf)|27.2|27.6|37.5|
|[Metric Learning With HORDE: High-Order Regularizer for Deep Embeddings (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jacob_Metric_Learning_With_HORDE_High-Order_Regularizer_for_Deep_Embeddings_ICCV_2019_paper.pdf)|55|72.2|N/A| 


#### Reported Precision@1 for the Triplet Loss
| Paper | CUB200 | Cars196 | SOP |
|-|-|-|-|
|[Deep Metric Learning via Lifted Structured Feature Embedding (CVPR 2016)](https://arxiv.org/pdf/1511.06452.pdf)|36.1|39.1|42.1|
|[Learning Deep Embeddings with Histogram Loss (NIPS 2016)](https://arxiv.org/pdf/1611.00822.pdf)|36.1|N/A|42.1|
|[Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NIPS 2016)](https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective.pdf)|43.3|53.84|53.32|
|[Hard-Aware Deeply Cascaded Embedding (ICCV 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf)|36.1|39.1|42.1|
|[Deep Metric Learning with Angular Loss (ICCV 2017)](https://arxiv.org/pdf/1708.01682.pdf)|42.2|45.5|56.5|
|[Deep Adversarial Metric Learning (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf)|35.9|45.1|53.9|
|[Deep Variational Metric Learning (ECCV 2018)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Hardness-Aware_Deep_Metric_Learning_CVPR_2019_paper.pdf)|39.8|58.5|54.9|
|[Deep Metric Learning with Hierarchical Triplet Loss (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ge_Deep_Metric_Learning_ECCV_2018_paper.pdf)|55.9|79.2|72.6|
|[Hardness-Aware Deep Metric Learning (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Hardness-Aware_Deep_Metric_Learning_CVPR_2019_paper.pdf)|35.9|45.1|53.9|
|[Deep Asymmetric Metric Learning via Rich Relationship Mining (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Deep_Asymmetric_Metric_Learning_via_Rich_Relationship_Mining_CVPR_2019_paper.pdf)|35.9|45.1|53.9|
|[Metric Learning With HORDE: High-Order Regularizer for Deep Embeddings (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jacob_Metric_Learning_With_HORDE_High-Order_Regularizer_for_Deep_Embeddings_ICCV_2019_paper.pdf)|50.5|65.2|N/A| 


## Frequently Asked Questions

#### Do you have slides that accompany the paper?
Slides are [here](https://docs.google.com/presentation/d/1KnLDFzMKLYlnMzMDc7wyKHVAh5dJ9z6Fs1qto4OqQFY/edit?usp=sharing){target=_blank}.


#### Isn't it unfair to fix the model, optimizer, learning rate, and embedding size?
Our goal was to compare algorithms fairly. To accomplish this, we used the same network, optimizer, learning rate, image transforms, and embedding dimensionality for each algorithm. There is no theoretical reason why changing any of these parameters would benefit one particular algorithm over the rest. If there is no theoretical reason, then we can only speculate, and if we add hyperparameters based on speculation, then the search space becomes too large to explore.

#### Why did you use BN-Inception?
We chose this architecture because it is commonly used in recent metric learning papers.

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


#### In Figure 2 (papers vs reality) why do you use Precision@1 instead of MAP@R?
None of the referenced papers report MAP@R. Since Figure 2a is meant to show reported results, we had to use a metric that was actually reported, i.e. Precision@1. We used the same metric for Figure 2b so that the two graphs could be compared directly side by side. But for the sake of completeness, here's Figure 2b using MAP@R:

![reality_over_time_mapr](mlrc_plots/reality_over_time_mapr.png)


## Reproducing results
### Download the experiment folder

1. Download [run.py and set the default flags](../index.md#getting-started)
2. Go to the [benchmark spreadsheet](https://docs.google.com/spreadsheets/d/1brUBishNxmld-KLDAJewIc43A4EVZk3gY6yKe8OIKbY/)
3. Find the experiment you want to reproduce, and click on the link in the "Config files" column.
4. You'll see 3 folders: one for CUB, one for Cars, and one for SOP. Open the folder for the dataset you want to train on.
5. Now you'll see several files and folders, one of which ends in "reproduction0". Download this folder. (It will include saved models. If you don't want to download the saved models, go into the folder and download just the "configs" folder.)

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
