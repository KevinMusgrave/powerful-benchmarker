import copy
from .base_factory import BaseFactory
from .model_factory import ModelFactory
from .transform_factory import TransformFactory
from collections import defaultdict
import torch
from ..utils import common_functions as c_f
import logging

class SplitManagerFactory(BaseFactory):
    def _create_general(self, split_manager_type):
        split_manager = self.get_split_manager(split_manager_type)

        chosen_dataset, original_dataset_params = self.getter.get("dataset", yaml_dict=self.api_parser.args.dataset, return_uninitialized=True)
        chosen_dataset = {k:chosen_dataset for k in split_manager.split_names}
        original_dataset_params = {k:original_dataset_params for k in split_manager.split_names}

        datasets = defaultdict(dict)
        for transform_type, T in self.api_parser.transforms.items():
            logging.info("{} transform: {}".format(transform_type, T))
            for split_name in split_manager.split_names:
                dataset_params = copy.deepcopy(original_dataset_params[split_name])
                dataset_params["transform"] = T
                if "root" not in dataset_params:
                    dataset_params["root"] = self.api_parser.args.dataset_root            
                datasets[transform_type][split_name] = chosen_dataset[split_name](**dataset_params)
        
        split_manager.create_split_schemes(datasets)
        return split_manager
               


    def get_split_manager(self, yaml_dict):
        split_manager, split_manager_params = self.getter.get("split_manager", yaml_dict=yaml_dict, return_uninitialized=True)
        split_manager_params = copy.deepcopy(split_manager_params)
        if c_f.check_init_arguments(split_manager, "model"):
            model_factory = ModelFactory(api_parser=self.api_parser, getter=self.getter)
            trunk_model = model_factory.create(named_specs=self.api_parser.args.models, subset="trunk")
            split_manager_params["model"] = torch.nn.DataParallel(trunk_model).to(self.api_parser.device)
        if "helper_split_manager" in split_manager_params:
            split_manager_params["helper_split_manager"] = self.get_split_manager(split_manager_params["helper_split_manager"])
        return split_manager(**split_manager_params)
