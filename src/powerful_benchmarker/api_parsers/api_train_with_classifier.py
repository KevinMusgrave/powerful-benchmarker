#! /usr/bin/env python3

from .base_api_parser import BaseAPIParser
from ..factories import ClassifierModelFactory

class APITrainWithClassifier(BaseAPIParser):
    def required_compatible_factories(self):
        return {"model": ClassifierModelFactory(getter=self.pytorch_getter)}

    def get_model(self, **kwargs):
        models = {}
        for k in self.factories["model"].creation_order(self.args.models):
            attr_key = "classifier" if k.startswith("classifier") else k
            models[k] = getattr(self, "get_{}".format(attr_key))(**kwargs)
        return models

    def get_classifier(self, **kwargs):
        return self.factories["model"].create(named_specs=self.args.models, subset="classifier", **self.all_kwargs("classifier", kwargs))

    def default_kwargs_classifier(self):
        return {"output_size": lambda: self.split_manager.get_num_labels("train", "train")}

    
