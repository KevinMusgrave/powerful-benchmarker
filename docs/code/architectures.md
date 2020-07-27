# Architectures

## ListOfModels
Turns a list of models into a single model. The specific behavior depends on the init parameters
```python
from powerful_benchmarker.architectures.misc_models import ListOfModels
ListOfModels(list_of_models, input_sizes=None, operation_before_concat=None):
```

### Parameters

- ```list_of_models```: A list of PyTorch models. The list will be converted into ```torch.nn.ModuleList(list_of_models)```.
- ```input_sizes```: A list of numbers, with the same length as ```list_of_models```. ```input_sizes[i]``` is the expected input size of ```list_of_models[i]```. Can also be left as ```None```.
- ```operation_before_concat```: A function that is applied to the output of ```list_of_models[i]``` before being concatenated with the output of the other models.

### Methods

#### forward

The standard PyTorch ```forward``` method, but the behavior differs depending on the value of ```self.input_sizes```.

If ```self.input_sizes``` is None, then each ```list_of_models[i]``` will receive the entire input ```x```.

If ```self.input_sizes``` is a list, then:

 - ```list_of_models[0]``` will receive ```x[:self.input_sizes[0]]```
 - ```list_of_models[1]``` will receive ```x[self.input_sizes[0]:self.input_sizes[1]]```
 - etc.


## MLP

A very simple multi layer perceptron.

```python
from powerful_benchmarker.architectures.misc_models import MLP
MLP(layer_sizes, final_relu=False)
```

### Parameters

- ```layer_sizes```: A list of numbers, where ```layer_sizes[0]``` is the size of the input, and ```layer_sizes[-1]``` is the size of the output. 
- ```final_relu```: If True, will apply ReLU to the final layer's input.