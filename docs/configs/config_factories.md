# config_factories

## factories
Factories determine how objects are constructed, based on parameters in the config files, and parameters generated within the code.

Default yaml:
```yaml
factories:
  model:
    ModelFactory: {}

  loss: 
    LossFactory: {}
                          
  miner: 
    MinerFactory: {}

  sampler:
    SamplerFactory: {}

  optimizer:
    OptimizerFactory: {}

  tester:
    TesterFactory: {}

  trainer:
    TrainerFactory: {}

  transform:
    TransformFactory: {}

  split_manager:
    SplitManagerFactory: {}

  record_keeper:
    RecordKeeperFactory: {}

  hook:
    HookFactory: {}

  aggregator:
    AggregatorFactory: {}

  ensemble:
    EnsembleFactory: {}
```

Example command line modification:
```bash
# Set the base_output_model_size manually
--factories {model~APPLY~2: {base_output_model_size: 1024}}
```