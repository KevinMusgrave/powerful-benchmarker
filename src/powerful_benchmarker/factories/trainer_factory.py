from .base_factory import BaseFactory
from easy_module_attribute_getter import utils as emag_utils
from .hook_factory import HookFactory

class TrainerFactory(BaseFactory):
    def _create_general(self, trainer_type, **kwargs):
        trainer, trainer_params = self.getter.get("trainer", yaml_dict=trainer_type, return_uninitialized=True)
        trainer_params = emag_utils.merge_two_dicts(trainer_params, kwargs)
        return trainer(**trainer_params)

