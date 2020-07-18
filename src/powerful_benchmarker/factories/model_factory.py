from ..utils import common_functions as c_f
from .. import architectures
import pytorch_metric_learning.utils.common_functions as pml_cf
from .base_factory import BaseFactory
import logging
import copy

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
        model_args = copy.deepcopy(model_args)
        if model == architectures.misc_models.MLP:
            if input_size:
                model_args["layer_sizes"].insert(0, input_size)
            if output_size:
                model_args["layer_sizes"].append(output_size)
        model = model(**model_args)
        logging.info("EMBEDDER MODEL %s"%model)
        return model

    def creation_order(self, specs):
        assert specs.keys() == {"trunk", "embedder"}, "The model names must be trunk and embedder"
        return ["trunk", "embedder"]

    def key_specific_kwargs(self, key):
        if key == "embedder":
            return {"input_size": self.base_model_output_size}
        return {}
        
