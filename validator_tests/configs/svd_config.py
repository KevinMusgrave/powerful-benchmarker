from pytorch_adapt.layers import BatchSpectralLoss, BNMLoss

from .base_config import BaseConfig, get_split_and_layer


class BSP(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator_args["k"] = int(self.validator_args["k"])
        self.layer = self.validator_args["layer"]
        self.validator = BatchSpectralLoss(k=self.validator_args["k"])

    def score(self, x, exp_config, device):
        features = get_split_and_layer(x, self.split, self.layer, device)
        return -self.validator(features)

    def expected_keys(self):
        return {"k", "split", "layer"}


class BNM(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.layer = self.validator_args["layer"]
        self.validator = BNMLoss()

    def score(self, x, exp_config, device):
        features = get_split_and_layer(x, self.split, self.layer, device)
        return -self.validator(features)

    def expected_keys(self):
        return {"split", "layer"}
