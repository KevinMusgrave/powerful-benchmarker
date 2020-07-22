#! /usr/bin/env python3
from .api_train_with_classifier import APITrainWithClassifier
from ..factories import ModelFactory, CascadedModelFactory

class APICascadedEmbeddings(APITrainWithClassifier):
    def required_compatible_factories(self):
        return {"model": CascadedModelFactory(getter=self.pytorch_getter)}

    def default_kwargs_transforms(self):
        basic_trunk_factory = ModelFactory(getter=self.pytorch_getter)
        return {"trunk_model": lambda: basic_trunk_factory.create(named_specs=self.args.models, subset="trunk")}

    def default_kwargs_trunk(self):
        return {"sample_input": lambda: self.split_manager.get_dataset("train", "train")[0]["data"].unsqueeze(0),
                "layers_to_extract": lambda: self.args.layers_to_extract}

    def default_kwargs_trainer(self):
        trainer_kwargs = super().default_kwargs_trainer()
        trainer_kwargs["embedding_sizes"] = lambda: self.factories["model"].all_embedding_sizes
        return trainer_kwargs