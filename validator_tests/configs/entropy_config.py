from pytorch_adapt.validators import EntropyValidator

from .base_config import BaseConfig, get_from_hdf5


class Entropy(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator = EntropyValidator(
            key_map={self.split: "target_train"},
        )

    def score(self, x, exp_config, device):
        logits = get_from_hdf5(x, device, f"inference/{self.split}/logits")
        return self.validator(**{self.split: {"logits": logits}})

    def expected_keys(self):
        return {"split"}
