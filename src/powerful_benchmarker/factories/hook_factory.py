from .base_factory import BaseFactory
import pytorch_metric_learning.utils.logging_presets as logging_presets
from easy_module_attribute_getter import utils as emag_utils
import logging
import torch

class HookFactory(BaseFactory):
    def create_hook_container(self, hook_container_type, **kwargs):
        hooks, hooks_params = self.getter.get("hook_container", yaml_dict=hook_container_type, return_uninitialized=True)
        hooks_params = emag_utils.merge_two_dicts(hooks_params, kwargs)
        return hooks(**hooks_params)

    def create_end_of_epoch_hook(self,
                                _,
                                hooks, 
                                split_manager, 
                                splits_to_eval, 
                                tester,
                                model_folder, 
                                save_interval, 
                                patience, 
                                collate_fn, 
                                eval_assertions):
        logging.info("Creating end_of_epoch_hook")
        dataset_dict = split_manager.get_dataset_dict("eval", inclusion_list=splits_to_eval)
        helper_hook = hooks.end_of_epoch_hook(tester=tester,
                                            dataset_dict=dataset_dict,
                                            model_folder=model_folder,
                                            test_interval=save_interval,
                                            patience=patience,
                                            test_collate_fn=collate_fn)
        def end_of_epoch_hook(trainer):
            torch.cuda.empty_cache()
            eval_assertions(dataset_dict)
            return helper_hook(trainer)

        return end_of_epoch_hook