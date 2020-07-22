from .base_factory import BaseFactory

class EnsembleFactory(BaseFactory):
    def _create_general(self, ensemble_type):
        return self.getter.get("ensemble", yaml_dict=ensemble_type)