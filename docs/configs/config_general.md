# config_general

## num_epochs_train
The maximum number of epochs to train for.

Default yaml:
```yaml
num_epochs_train: 1000
```

Command line:
```bash
--num_epochs_train 1000
```

## iterations_per_epoch
If ```null```:

* 1 epoch = 1 pass through the dataloader iterator. If ```sampler=None```, then 1 pass through the iterator is 1 pass through the dataset. 
* If you use a sampler, then 1 pass through the iterator is 1 pass through the iterable returned by the sampler.

If an integer, then an epoch consists of this many iterations.

* Why have this option? For samplers like ```MPerClassSampler``` or some offline mining method, the iterable returned might be very long or very short etc, and might not be related to the length of the dataset. The length of the epoch might vary each time the sampler creates a new iterable. In these cases, it can be useful to specify ```iterations_per_epoch``` so that each "epoch" is just a fixed number of iterations. The definition of epoch matters because there's various things like LR schedulers and hooks that depend on an epoch ending.

Default yaml:
```yaml
iterations_per_epoch: 100
```

Command line:
```bash
--iterations_per_epoch 100
```

## save_interval
Models will be evaluated and saved every ```save_interval``` epochs.

Default yaml:
```yaml
save_interval: 2
```

Command line:
```bash
--save_interval 2
```

## check_untrained_accuracy
If ```True```, then the tester will compute accuracy for the initial trunk (epoch -1) and initial trunk + embedder (epoch 0). Otherwise, these will be skipped.

Default yaml:
```yaml
check_untrained_accuracy: True
```

Command line:
```bash
--check_untrained_accuracy True
```

## skip_eval_if_already_done
If ```True```, then the tester will skip evaluation if a split/epoch has already been logged in the log files. If ```False```, then the tester will evaluate a split/epoch regardless of whether it has already been done in the past. Previous logs will be preserved, hence the logs will contain duplicate results, and the most recent version for any split/epoch will be considered the "official" value for that split/epoch.

Default yaml:
```yaml
skip_eval_if_already_done: True
```

Command line:
```bash
--skip_eval_if_already_done True
```

## skip_meta_eval_if_already_done
The same as ```skip_eval_if_already_done```, but for meta evaluation.

Default yaml:
```yaml
skip_meta_eval_if_already_done: True
```

Command line:
```bash
--skip_meta_eval_if_already_done True
```