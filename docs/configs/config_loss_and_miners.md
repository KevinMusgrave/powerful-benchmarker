# config_loss_and_miners

## loss_funcs
An object mapping from strings to loss classes. The strings should match the loss names used by your trainer.

Default yaml:
```yaml
loss_funcs:
  metric_loss: 
    ContrastiveLoss:
```

Command line:
```bash
--loss_funcs {metric_loss: {ContrastiveLoss: {}}}
```

## sampler
The sampler class, or ```null``` if you want random sampling.

Default yaml:
```yaml
sampler:
  MPerClassSampler:
    m: 4
```

Command line:
```bash
--sampler {MPerClassSampler: {m: 4}}
```

## mining_funcs
An object mapping from strings to mining classes. The strings should match the mining names used by your trainer.

Default yaml:
```yaml
mining_funcs: {}
```

A non-empty yaml example:
```yaml
mining_funcs:
  tuple_miner:
    MultiSimilarityMiner:
       epsilon: 0.1
```

Command line:
```bash
--mining_funcs {tuple_miner: {MultiSimilarityMiner: {epsilon: 0.1}}}
```