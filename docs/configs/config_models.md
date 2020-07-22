# config_models

## models
The models take in input (like images, text etc.) and output embeddings. There is no specific requires about what the structure of the trunk and embedder. The only requirement is that the trunk's output can be fed into the embedder. For example, if you want to use the ```bninception``` model, but don't want to append any layers after it, you can set embedder to ```Identity```. This will make the embedder's output equal to its input.

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

Example command line modification:
```bash
# Set embedder to Identity.
--models {embedder~OVERRIDE~: {Identity: {}}} \
# You'll need to delete the embedder_optimizer, because Identity() has no parameters 
--optimizers {embedder_optimizer~DELETE~: null}
```
