from .base_factory import BaseFactory
from easy_module_attribute_getter import utils as emag_utils
from .hook_factory import HookFactory

class TrainerFactory(BaseFactory):
    def _create_general(self, trainer_type):
        hook_factory = HookFactory(api_parser=self.api_parser, getter=self.getter)
        trainer, trainer_params = self.getter.get("trainer", yaml_dict=trainer_type, return_uninitialized=True)
        x = self.api_parser
        other_args = {
            "models": x.models,
            "optimizers": x.optimizers,
            "sampler": x.sampler,
            "collate_fn": x.split_manager.collate_fn,
            "loss_funcs": x.loss_funcs,
            "mining_funcs": x.mining_funcs,
            "dataset": x.split_manager.get_dataset("train", "train", log_split_details=True),
            "data_device": x.device,
            "lr_schedulers": x.lr_schedulers,
            "gradient_clippers": x.gradient_clippers,
            "data_and_label_getter": x.split_manager.data_and_label_getter,
            "dataset_labels": list(x.split_manager.get_label_set("train", "train")),
            "end_of_iteration_hook": x.hooks.end_of_iteration_hook,
            "end_of_epoch_hook": hook_factory.create_end_of_epoch_hook()
        }
        trainer_params = emag_utils.merge_two_dicts(trainer_params, other_args)
        return trainer(**trainer_params)

