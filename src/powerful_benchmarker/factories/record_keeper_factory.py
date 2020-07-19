from .base_factory import BaseFactory
import pytorch_metric_learning.utils.logging_presets as logging_presets

class RecordKeeperFactory(BaseFactory):
    def _create_general(self, record_keeper_type):
        record_keeper, _, _ = logging_presets.get_record_keeper(**record_keeper_type)
        return record_keeper