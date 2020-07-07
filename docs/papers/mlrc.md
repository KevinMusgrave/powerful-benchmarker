# A Metric Learning Reality Check

This page contains additional information for the [ECCV 2020 paper](https://arxiv.org/abs/2003.08505) by Musgrave et al.

## Frequently Asked Questions

#### Isn't it unfair to fix the model, optimizer, learning rate, and embedding size?

#### Why weren't more hard-mining methods evaluated?

#### Why was the batch size set to 32 for most of the results?

#### What is the difference between MAP@R and MAP?

#### For the contrastive loss, why is the optimal positive margin a negative value?

- A negative value should be equivalent to a margin of 0, because the distance between positive pairs cannot be negative, and the margin does not contribute to the gradient. So allowing the hyperparameter optimization to explore negative margins was unnecesary.


## Optimization plots




