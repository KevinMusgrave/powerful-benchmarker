from pytorch_adapt.layers import BatchSpectralLoss, BNMLoss

from .base_config import BaseConfig, get_from_hdf5


class BSP(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator_args["k"] = int(self.validator_args["k"])
        self.validator = BatchSpectralLoss(k=self.validator_args["k"])

    def score(self, x, exp_config, device):
        features = get_from_hdf5(x, device, f"inference/{self.split}/{self.layer}")
        return -self.validator(features)

    def expected_keys(self):
        return {"k", "split", "layer"}


class BNM(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator = BNMLoss()

    def score(self, x, exp_config, device):
        features = get_from_hdf5(x, device, f"inference/{self.split}/{self.layer}")
        return -self.validator(features)

    def expected_keys(self):
        return {"split", "layer"}
