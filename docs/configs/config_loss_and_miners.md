# config_loss_and_miners

## loss_funcs
The loss functions are given embeddings and labels, and output a value on which back propagation can be performed. This config option is a mapping from strings to loss classes. The strings should match the loss names used by your trainer. 

Default yaml:
```yaml
loss_funcs:
  metric_loss: 
    ContrastiveLoss:
```

Example command line modification:
```bash
# Use a different loss function
--loss_funcs {metric_loss~OVERRIDE~: {MultiSimilarityLoss: {alpha: 0.1, beta: 40, base: 0.5}}}
```

## sampler
The sampler is passed to the PyTorch dataloader, and determines how batches are formed. Use ```{}``` if you want random sampling.

Default yaml:
```yaml
sampler:
  MPerClassSampler:
    m: 4
```

Example command line modification:
```bash
# Use random sampling
--sampler~OVERRIDE~ {}
```

## mining_funcs
Mining functions determine the best tuples to train on, within an arbitrarily formed batch. This config option is a mapping from strings to miner classes. The strings should match the miner names used by your trainer.

Default yaml:
```yaml
mining_funcs: {}
```

Example command line modification:
```bash
# Use a miner
--mining_funcs {tuple_miner: {MultiSimilarityMiner: {epsilon: 0.1}}}
```