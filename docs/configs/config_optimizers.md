# config_optimizers

## optimizers
Optimizers determine how your model weights are updated. This config option maps from strings to optimizer classes. Each string should have the form ```<model_name>_optimizer```.

Default yaml:
```yaml
optimizers:
  trunk_optimizer:
    RMSprop:
      lr: 0.000001
      weight_decay: 0.0001
      momentum: 0.9
  embedder_optimizer:
    RMSprop:
      lr: 0.000001
      weight_decay: 0.0001
      momentum: 0.9
```

Example command line modification:
```bash
# Change the learning rate to 0.01, for both the trunk and embedder
--optimizers~APPLY~3 {lr: 0.01}
```
