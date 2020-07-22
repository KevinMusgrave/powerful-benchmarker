from ..utils import common_functions as c_f
import pytorch_metric_learning.utils.common_functions as pml_cf
from .base_factory import BaseFactory
import logging
import copy
from .. import architectures as arch
import torch

class ModelFactory(BaseFactory):
    def __init__(self, base_model_output_size=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model_output_size = base_model_output_size

    def create_trunk(self, model_type):
        model = self.getter.get("model", yaml_dict=model_type)
        try:
            self.base_model_output_size = c_f.get_last_linear(model).in_features
        except AttributeError:
            assert self.base_model_output_size is not None, "The base model output size could not be inferred. Please set it manually"
        c_f.set_last_linear(model, pml_cf.Identity())
        return model

    def create_embedder(self, model_type, input_size=None, output_size=None):
        model, model_args = self.getter.get("model", yaml_dict=model_type, return_uninitialized=True)
        if model == arch.misc_models.MLP:
            model_args = self.modify_mlp_args(model, model_args, input_size, output_size)
        model = model(**model_args)
        logging.info("EMBEDDER MODEL %s"%model)
        return model

    def creation_order(self, specs):
        assert set(self.required_models()).issubset(set(specs.keys())), "The model names must include all of {}".format(self.required_models())
        order = self._creation_order()
        for k in self.optional_models():
            if k not in specs:
                logging.warning("The optional model, '{}', is not specified for {}".format(k, self.__class__.__name__))
                order.remove(k)
        return order

    def key_specific_kwargs(self, key):
        if key == "embedder":
            return {"input_size": self.base_model_output_size}
        return {}

    def modify_mlp_args(self, model, model_args, input_size, output_size):
        model_args = copy.deepcopy(model_args)
        if input_size:
            model_args["layer_sizes"].insert(0, input_size)
        if output_size:
            model_args["layer_sizes"].append(output_size)
        return model_args

    def _creation_order(self):
        return ["trunk", "embedder"]

    def required_models(self):
        return ["trunk", "embedder"]

    def optional_models(self):
        return []


class ClassifierModelFactory(ModelFactory):
    def __init__(self, embedder_output_size=None, **kwargs):
        super().__init__(**kwargs)
        self.embedder_output_size = embedder_output_size

    def create_embedder(self, model_type, input_size=None, output_size=None):
        model = super().create_embedder(model_type, input_size, output_size)
        try:
            self.embedder_output_size = c_f.get_last_linear(model).out_features
        except AttributeError:
            assert self.embedder_output_size is not None, "The embedder output size could not be inferred. Please set it manually"
        return model

    def create_classifier(self, model_type, output_size):
        model, model_args = self.getter.get("model", yaml_dict=model_type, return_uninitialized=True)
        model_args = self.modify_mlp_args(model, model_args, self.embedder_output_size, output_size)
        model = model(**model_args)
        logging.info("CLASSIFIER MODEL %s"%model)
        return model

    def _creation_order(self):
        return ["trunk", "embedder", "classifier"]

    def optional_models(self):
        return ["classifier"]



class GeneratorModelFactory(ClassifierModelFactory):
    def create_generator(self, model_type):
        model, model_args = self.getter.get("model", yaml_dict=model_type, return_uninitialized=True)
        model_args = copy.deepcopy(model_args)
        model_args["layer_sizes"] = [x*self.base_model_output_size for x in model_args["layer_sizes"]]
        model = model(**model_args)
        logging.info("GENERATOR MODEL %s"%model)
        return model

    def _creation_order(self):
        return ["trunk", "embedder", "generator", "classifier"]


class CascadedModelFactory(ClassifierModelFactory):
    def create_trunk(self, model_type, sample_input, layers_to_extract):
        model = super().create_trunk(model_type)
        model_name = c_f.first_key_of_dict(model_type)
        model = arch.misc_models.LayerExtractor(
            model,
            layers_to_extract,
            self.get_skip_layer_names(model_name),
            self.get_insert_functions(model_name),
        ).eval()
        with torch.no_grad():
            _, self.base_model_output_size = model.layer_by_layer(sample_input, return_layer_sizes=True)
        return model

    def create_embedder(self, model_type, input_size=None, output_size=None):
        embedders = []
        for i in input_size:
            embedders.append(super().create_embedder(model_type, input_size=i, output_size=output_size))
        self.all_embedding_sizes = [c_f.get_last_linear(embedders[0]).out_features] * len(embedders)
        model = arch.misc_models.ListOfModels(embedders, input_size)
        return model

    def key_specific_kwargs(self, key):
        if key == "embedder":
            return {"input_size": self.base_model_output_size}
        return {}

    def get_skip_layer_names(self, model_name):
        return {"inception_v3": "AuxLogits"}[model_name]

    def get_insert_functions(self, model_name):
        return {
            "inception_v3": {
                "Conv2d_2b_3x3": [torch.nn.MaxPool2d(3, stride=2)],
                "Conv2d_4a_3x3": [torch.nn.MaxPool2d(3, stride=2)],
            }
        }[model_name]