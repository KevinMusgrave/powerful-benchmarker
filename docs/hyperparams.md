# Hyperparameter optimization

## Bayesian optimization

### Syntax
To tune hyperparameters using bayesian optimization:

1. In your config files or at the command line, append ```~BAYESIAN~``` to any parameter that you want to tune, followed by a lower and upper bound in square brackets. Use ```~LOG_BAYESIAN~``` for log-scaled parameters, and ```~INT_BAYESIAN~``` for integer parameters.
2. Specify the number of bayesian optimization iterations with the ```--bayes_opt_iters``` command line flag.

Here is an example script which uses bayesian optimization to tune 3 hyperparameters for the multi similarity loss.
```bash
python run.py --bayes_opt_iters 50 \
--loss_funcs~OVERRIDE~ \
{metric_loss: {MultiSimilarityLoss: {\
alpha~LOG_BAYESIAN~: [0.01, 100], \
beta~LOG_BAYESIAN~: [0.01, 100], \
base~BAYESIAN~: [0, 1]}}} \
--experiment_name cub_bayes_opt \
```

### Resume optimization
If you stop and want to resume bayesian optimization, simply run ```run.py``` with the same ```experiment_name``` you were using before. 

### Change optimization bounds
You can change the optimization bounds when resuming, by either changing the bounds in the config files or at the command line. The command line is preferable, because any config diffs will be recorded (just like in [regular experiments](index.md#keep-track-of-changes)). If you're using the command line, make sure to also use the ```--merge_argparse_when_resuming``` flag.

### Run reproductions
You can run a number of reproductions for the best parameters, so that you can obtain a confidence interval for your results. Use the ```reproductions``` flag, and pass in the number of reproductions you want to perform at the end of bayesian optimization.
```bash
python run.py --bayes_opt_iters 50 --reproductions 10 \
--experiment_name cub_bayes_opt \
```
