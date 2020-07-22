# config_eval

## tester
The tester computes the accuracy of your model.

Default yaml:
```yaml
tester:
  GlobalEmbeddingSpaceTester:
    reference_set: compared_to_self
    normalize_embeddings: True
    use_trunk_output: False
    batch_size: 32
    dataloader_num_workers: 2
    pca: null
    accuracy_calculator:
      AccuracyCalculator:
    label_hierarchy_level: 0
```
Example command line modification:
```bash
# Change batch size to 256 and don't normalize embeddings
--tester~APPLY~2 {batch_size: 256, normalize_embeddings: False}
```

## aggregator
The aggregator takes the accuracies from all the cross-validation models, and returns a single number to represent the overall performance.

Default yaml:
```yaml
aggregator:
  MeanAggregator:
```

Example command line modification:
```bash
# Use your own custom aggregator
--aggregator~OVERRIDE~ {YourCustomAggregator: {}}
```

## ensemble
The ensemble combines the cross-validation models into a single model.

Default yaml:
```yaml
ensemble:
  ConcatenateEmbeddings:
    normalize_embeddings: True
    use_trunk_output: False
```

## hook_container
The hook container contains end-of-testing, end-of-epoch, and end-of-iteration hooks. It also contains a record keeper, for writing and reading to database files.

Default yaml:
```yaml
hook_container:
  HookContainer:
    primary_metric: mean_average_precision_at_r
    validation_split_name: val
```

Example command line modification:
```bash
# Change the primary metric to precision_at_1
--hook_container~APPLY~2 {primary_metric: precision_at_1}
```

