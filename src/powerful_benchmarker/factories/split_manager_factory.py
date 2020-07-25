import copy
from .base_factory import BaseFactory
from collections import defaultdict
import logging

class SplitManagerFactory(BaseFactory):
    def _create_general(self, split_manager_type, dataset, original_dataset_params, transforms, dataset_root):
        split_manager = self.get_split_manager(split_manager_type)

        dataset = {k:dataset for k in split_manager.split_names}
        original_dataset_params = {k:original_dataset_params for k in split_manager.split_names}

        datasets = defaultdict(dict)
        dataset_count = 0
        for transform_type, T in transforms.items():
            logging.info("{} transform: {}".format(transform_type, T))
            for split_name in split_manager.split_names:
                dataset_params = copy.deepcopy(original_dataset_params[split_name])
                dataset_params["transform"] = T
                if "root" not in dataset_params:
                    dataset_params["root"] = dataset_root  
                if dataset_count > 0:
                    dataset_params["download"] = False         
                datasets[transform_type][split_name] = dataset[split_name](**dataset_params)
                dataset_count += 1
        
        split_manager.create_split_schemes(datasets)
        return split_manager
               


    def get_split_manager(self, yaml_dict):
        split_manager, split_manager_params = self.getter.get("split_manager", yaml_dict=yaml_dict, return_uninitialized=True)
        split_manager_params = copy.deepcopy(split_manager_params)
        if "helper_split_manager" in split_manager_params:
            split_manager_params["helper_split_manager"] = self.get_split_manager(split_manager_params["helper_split_manager"])
        return split_manager(**split_manager_params)
