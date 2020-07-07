# config_optimizers

## optimizers
An object mapping from strings to optimizer objects. The strings should have the form ```<model_name>_optimizer```.

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

Command line:
```bash
--optimizers {trunk_optimizer: {RMSprop: {lr: 0.000001, weight_decay: 0.0001, momentum: 0.9}}, embedder_optimizer: {RMSprop: {lr: 0.000001, weight_decay: 0.0001, momentum: 0.9}}}
```
