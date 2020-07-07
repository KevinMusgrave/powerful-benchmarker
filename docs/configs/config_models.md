# config_models

## models
An object mapping from strings to the models that create embeddings.

Default yaml:
```yaml
models:
  trunk:
    bninception:
      pretrained: imagenet
  embedder:
    MLP:
      layer_sizes:
        - 128
```

Command line:
```bash
--models {trunk: {bninception: {pretrained: imagenet}}, embedder: {MLP: {layer_size: [128]}}}
```
