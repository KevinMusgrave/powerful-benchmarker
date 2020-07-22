from .base_factory import BaseFactory
from easy_module_attribute_getter import utils as emag_utils
import copy

class TesterFactory(BaseFactory):
    def _create_general(self, tester_type, **kwargs):
        tester, tester_params = self.getter.get("tester", yaml_dict=tester_type, return_uninitialized=True)
        tester_params = copy.deepcopy(tester_params)
        tester_params["accuracy_calculator"] = self.getter.get("accuracy_calculator", yaml_dict=tester_params["accuracy_calculator"])
        tester_params = emag_utils.merge_two_dicts(tester_params, kwargs)
        return tester(**tester_params)
               