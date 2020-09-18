# Ensembles

Ensembles take multiple models and combine them into a single model.

## BaseEnsemble
```python
from powerful_benchmarker.ensembles import BaseEnsemble
BaseEnsemble(normalize_embeddings=True, use_trunk_output=False)
```

### Parameters

 - ```normalize_embeddings```: Perform L2 normalization if True. The specific details are determined by the child class.
 - ```use_trunk_output```: Use the output of the trunk of the ensemble. The specific details are determined by the child class.

### Methods

#### get_list_of_models

Loads models given a list of split scheme folders, and returns a list containing the loaded models.

```python
get_list_of_models(model_factory, model_args, model_name, factory_kwargs, split_folders, device)
```

#### create_ensemble_model

Returns a single trunk and embedder, given a list of trunks and embedders. Must be implemented by the child class.

```python
create_ensemble_model(list_of_trunks, list_of_embedders)
```

## ConcatenateEmbeddings

Returns a trunk that outputs the concatenation of multiple trunk models.

Returns an embedder that outputs the concatenation of multiple embedder models.

The trunk's output can be passed into the embedder.

```python
from powerful_benchmarker.ensembles import ConcatenateEmbeddings
ConcatenateEmbeddings(**kwargs)
```
