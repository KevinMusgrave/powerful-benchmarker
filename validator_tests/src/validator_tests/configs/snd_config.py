from pytorch_adapt.validators import SNDValidator

from .base_config import BaseConfig, get_split_and_layer


class SND(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator_args["T"] = float(self.validator_args["T"])
        self.layer = self.validator_args["layer"]
        self.validator = SNDValidator(
            key_map={self.split: "target_train"},
            layer=self.layer,
            T=self.validator_args["T"],
        )

    def score(self, x, exp_config, device):
        features = get_split_and_layer(x, self.split, self.layer, device)
        return self.validator(**{self.split: {self.layer: features}})
