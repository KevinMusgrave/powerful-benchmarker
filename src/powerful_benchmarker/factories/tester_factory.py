import copy
from .base_factory import BaseFactory
from easy_module_attribute_getter import utils as emag_utils

class TesterFactory(BaseFactory):
    def _create_general(self, tester_type, device, data_and_label_getter, end_of_testing_hook):
        tester, tester_params = self.getter.get("tester", yaml_dict=tester_type, return_uninitialized=True)
        accuracy_calculator = self.getter.get("accuracy_calculator", yaml_dict=tester_params["accuracy_calculator"])
        other_args = {"data_device": device,
                    "data_and_label_getter": data_and_label_getter,
                    "end_of_testing_hook": end_of_testing_hook,
                    "accuracy_calculator": accuracy_calculator}
        tester_params = emag_utils.merge_two_dicts(tester_params, other_args)
        return tester(**tester_params)
               