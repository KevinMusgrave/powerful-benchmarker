from pytorch_adapt.layers import MCCLoss

from .base_config import BaseConfig, get_split_and_layer


class MCC(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator_args["T"] = float(self.validator_args["T"])
        self.validator = MCCLoss(T=self.validator_args["T"])

    def score(self, x, exp_config, device):
        logits = get_split_and_layer(x, self.split, "logits", device)
        return -self.validator(logits).item()

    def expected_keys(self):
        return {"split", "T"}
