from .base_factory import BaseFactory

class AggregatorFactory(BaseFactory):
    def _create_general(self, aggregator_type):
        return self.getter.get("aggregator", yaml_dict=aggregator_type)