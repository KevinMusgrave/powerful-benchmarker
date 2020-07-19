import copy
from .base_factory import BaseFactory

class MinerFactory(BaseFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from .loss_factory import LossFactory
        self.loss_factory = LossFactory

    def _create_general(self, miner_type):
        miner, miner_params = self.getter.get("miner", yaml_dict=miner_type, return_uninitialized=True)
        miner_params = copy.deepcopy(miner_params)
        if "loss" in miner_params: 
            self.loss_factory(getter=self.getter)
            miner_params["loss"] = self.loss_factory.create(miner_params["loss"])
        if "miner" in miner_params: miner_params["miner"] = self.create(miner_params["miner"])
        return miner(**miner_params)