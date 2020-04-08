#! /usr/bin/env python3
import torch
from .. import architectures as arch
from .api_train_with_classifier import APIMaybeExtendTrainWithClassifier
import numpy as np
from ..utils import common_functions as c_f
from pytorch_metric_learning.utils import common_functions as pml_c_f
import logging

class APICascadedEmbeddings(APIMaybeExtendTrainWithClassifier):
    def get_trainer_kwargs(self):
        trainer_kwargs = self.inheriter.get_trainer_kwargs()
        trainer_kwargs["embedding_sizes"] = self.all_embedding_sizes
        return trainer_kwargs

    def get_classifier_model(self, model_type, output_size):
        input_size = c_f.get_last_linear(self.models["embedder"].list_of_models[0]).out_features
        return super().get_embedder_model(model_type, input_size, output_size)

    def get_embedder_model(self, model_type, input_size=None, output_size=None):
        embedders = []
        for i in input_size:
            embedders.append(self.inheriter.get_embedder_model(model_type, input_size=i, output_size=output_size))
        self.all_embedding_sizes = [c_f.get_last_linear(embedders[0]).out_features] * len(embedders)
        model = arch.misc_models.ListOfModels(embedders, input_size)
        return model

    def get_trunk_model(self, model_type):
        model = self.inheriter.get_trunk_model(model_type)
        logging.info("GETTING SAMPLE DATA TO DETERMINE MLP SIZE")
        self.set_transforms()
        sample_input = self.split_manager.dataset[0]["data"].unsqueeze(0)
        (model_name, _), = model_type.items()
        model = arch.misc_models.LayerExtractor(
            model,
            self.args.layers_to_extract,
            self.get_skip_layer_names(model_name),
            self.get_insert_functions(model_name),
        ).eval()
        with torch.no_grad():
            _, self.base_model_output_size = model.layer_by_layer(sample_input, return_layer_sizes=True)
        return model

    def get_skip_layer_names(self, model_name):
        return {"inception_v3": "AuxLogits"}[model_name]

    def get_insert_functions(self, model_name):
        return {
            "inception_v3": {
                "Conv2d_2b_3x3": [torch.nn.MaxPool2d(3, stride=2)],
                "Conv2d_4a_3x3": [torch.nn.MaxPool2d(3, stride=2)],
            }
        }[model_name]