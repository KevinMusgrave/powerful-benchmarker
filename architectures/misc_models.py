import torch
import torch.nn as nn


class LayerExtractor(nn.Module):
    def __init__(self, convnet, keep_layers, skip_layers, insert_functions):
        super().__init__()
        self.convnet = convnet
        self.keep_layers = keep_layers
        self.skip_layers = skip_layers
        self.insert_functions = insert_functions
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        for k in ["mean", "std", "input_space", "input_range"]:
            setattr(self, k, getattr(convnet, k, None))

    def forward(self, x):
        return self.layer_by_layer(x)

    def layer_by_layer(self, x, return_layer_sizes=False):
        outputs = []
        for layer_name, layer in self.convnet.named_children():
            if layer_name in self.skip_layers:
                continue
            x = layer(x)
            if layer_name in self.insert_functions:
                for zzz in self.insert_functions[layer_name]:
                    x = zzz(x)
            if layer_name in self.keep_layers:
                pooled_x = self.pooler(x).view(x.size(0), -1)
                outputs.append(pooled_x)
        output = torch.cat(outputs, dim=-1)
        if return_layer_sizes:
            return output, [x.size(-1) for x in outputs]
        return output



class ListOfModels(nn.Module):
    def __init__(self, list_of_models, input_sizes=None, operation_before_concat=None):
        super().__init__()
        self.list_of_models = nn.ModuleList(list_of_models)
        self.input_sizes = input_sizes
        self.operation_before_concat = (lambda x: x) if not operation_before_concat else operation_before_concat
        for k in ["mean", "std", "input_space", "input_range"]:
            setattr(self, k, getattr(list_of_models[0], k, None))

    def forward(self, x):
        outputs = []
        if self.input_sizes is None:
            for m in self.list_of_models:
                curr_output = self.operation_before_concat(m(x))
                outputs.append(curr_output)
        else:
            s = 0
            for i, y in enumerate(self.input_sizes):
                curr_input = x[:, s : s + y]
                curr_output = self.operation_before_concat(self.list_of_models[i](curr_input))
                outputs.append(curr_output)
                s += y
        return torch.cat(outputs, dim=-1)


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

