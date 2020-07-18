from .base_factory import BaseFactory
import pytorch_metric_learning.utils.logging_presets as logging_presets
from easy_module_attribute_getter import utils as emag_utils
import logging
import torch

class HookFactory(BaseFactory):
    def create_hook_container(self, hook_container_type, **kwargs):
        hooks, hooks_params = self.getter.get("hook_container", yaml_dict=hook_container_type, return_uninitialized=True)
        if "record_keeper" not in kwargs:
            hooks_params["record_keeper"] = self.api_parser.record_keeper
        hooks_params = emag_utils.merge_two_dicts(hooks_params, kwargs)
        return hooks(**hooks_params)

    def create_end_of_epoch_hook(self):
        logging.info("Creating end_of_epoch_hook kwargs")
        dataset_dict = self.api_parser.split_manager.get_dataset_dict("eval", inclusion_list=self.api_parser.args.splits_to_eval)
        helper_hook = self.api_parser.hooks.end_of_epoch_hook(tester=self.api_parser.tester,
                                                    dataset_dict=dataset_dict,
                                                    model_folder=self.api_parser.model_folder,
                                                    test_interval=self.api_parser.args.save_interval,
                                                    patience=self.api_parser.args.patience,
                                                    test_collate_fn=self.api_parser.split_manager.collate_fn)
        def end_of_epoch_hook(trainer):
            torch.cuda.empty_cache()
            self.api_parser.eval_assertions(dataset_dict)
            return helper_hook(trainer)

        return end_of_epoch_hook