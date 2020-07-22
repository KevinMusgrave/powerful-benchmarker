#! /usr/bin/env python3

from .api_train_with_classifier import APITrainWithClassifier
from ..factories import GeneratorModelFactory

class APIDeepAdversarialMetricLearning(APITrainWithClassifier):
    def required_compatible_factories(self):
        return {"model": GeneratorModelFactory(getter=self.pytorch_getter)}

    def get_generator(self, **kwargs):
        return self.factories["model"].create(named_specs=self.args.models, subset="generator", **self.all_kwargs("generator", kwargs))
