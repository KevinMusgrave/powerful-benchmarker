# Aggregators

Given the accuracies of multiple models, an aggregator will return a single value representing the total score.

## BaseAggregator

The base aggregator class.

```python
from powerful_benchmarker.aggregators import BaseAggregator
BaseAggregator()
```

### Methods

#### update_accuracies
Updates the internal state with the accuracy for a particular split scheme and splits.
```python
update_accuracies(split_scheme_name, splits_to_eval, hooks, tester)
```

#### record_accuracies
Saves the internal state to the record keeper (CSV, SQLite, and tensorboard).
```python
record_accuracies(splits_to_eval, meta_record_keeper, hooks, tester)
```

#### get_accuracy_and_standard_error
If more than one split scheme is used, then the aggregate accuracy and standard error of the mean is returned. Otherwise, just the aggregate accuracy is returned.

```python
get_accuracy_and_standard_error(hooks, tester, meta_record_keeper, num_split_schemes, split_name)
```

#### get_aggregate_performance

Must be implemented by the child class.

```python
get_aggregate_performance(accuracy_per_split)
```


## MeanAggregator

Returns the mean accuracy of multiple models.

```python
from powerful_benchmarker.aggregators import MeanAggregator
MeanAggregator()
```
