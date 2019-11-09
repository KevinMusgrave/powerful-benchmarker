#! /usr/bin/env python3

from .base_api_parser import BaseAPIParser
from utils import common_functions as c_f

class APIParserTrainWithClassifier(BaseAPIParser):
    def model_getter_dict(self):
        getter_dict = super().model_getter_dict()
        getter_dict["classifier"] = lambda model_type: self.get_embedder_model(
            model_type,
            c_f.get_last_linear(self.models["embedder"]).out_features,
            self.split_manager.get_num_labels(self.args.label_hierarchy_level),
        )
        return getter_dict



class APIMaybeExtendTrainWithClassifier(APIParserTrainWithClassifier):
    def __init__(self, args):
        super().__init__(args)
        loss_weights = getattr(self.args, "loss_weights", None)
        if loss_weights and "classifier_loss" in loss_weights:
            self.inheriter = super()
        else:
            self.inheriter = super(APIParserTrainWithClassifier, self)

    def model_getter_dict(self):
        return self.inheriter.model_getter_dict()