#! /usr/bin/env python3

from .base_api_parser import BaseAPIParser
from ..utils import common_functions as c_f
import logging

class APITrainWithClassifier(BaseAPIParser):

    def get_classifier_model(self, model_type, output_size):
        input_size = c_f.get_last_linear(self.models["embedder"]).out_features
        return super().get_embedder_model(model_type, input_size, output_size)

    def model_getter_dict(self):
        logging.info("Setting dataset so that num labels can be determined")
        self.split_manager.set_curr_split("train", is_training=True, log_split_details=True)        
        getter_dict = super().model_getter_dict()
        classifer_model_names = [x for x in list(self.args.models.keys()) if x.startswith("classifier")]
        for k in classifer_model_names:
            getter_dict[k] = lambda model_type: self.get_classifier_model(
                model_type,
                self.split_manager.get_num_labels(),
            )
        return getter_dict



class APIMaybeExtendTrainWithClassifier(APITrainWithClassifier):
    def __init__(self, args, *positional_args, **kwargs):
        super().__init__(args, *positional_args, **kwargs)
        model_names = list(self.args.models.keys())
        if model_names and any(x.startswith("classifier") for x in model_names):
            self.inheriter = super()
        else:
            self.inheriter = super(APITrainWithClassifier, self)

    def model_getter_dict(self):
        return self.inheriter.model_getter_dict()