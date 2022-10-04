import torch
from pytorch_adapt.layers import BatchSpectralLoss
from pytorch_adapt.validators import BNMValidator

from .base_config import BaseConfig, get_split_and_layer


class BSP(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator_args["k"] = int(self.validator_args["k"])
        self.layer = self.validator_args["layer"]
        self.validator = BatchSpectralLoss(k=self.validator_args["k"])

    def score(self, x, exp_config, device):
        features = get_split_and_layer(x, self.split, self.layer, device)
        try:
            return -self.validator(features).item()
        except RuntimeError as e:
            if "svd_cuda" in str(e):
                return float("nan")
            raise

    def expected_keys(self):
        return {"k", "split", "layer"}


class BNM(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.layer = self.validator_args["layer"]
        self.validator = BNMValidator(
            key_map={self.split: "target_train"},
        )

    def score(self, x, exp_config, device):
        features = get_split_and_layer(x, self.split, self.layer, device)
        return self.validator(**{self.split: {self.layer: features}})

    def expected_keys(self):
        return {"split", "layer"}


# from https://github.com/cuishuhao/BNM
# TODO: move to pytorch-adapt
def FBNM_loss(X):
    X = torch.nn.functional.softmax(X, dim=1)
    list_svd, _ = torch.sort(
        torch.sqrt(torch.sum(torch.pow(X, 2), dim=0)), descending=True
    )
    nums = min(X.shape[0], X.shape[1])
    return -torch.sum(list_svd[:nums])


class FBNM(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.layer = self.validator_args["layer"]

    def score(self, x, exp_config, device):
        features = get_split_and_layer(x, self.split, self.layer, device)
        return -FBNM_loss(features).item()

    def expected_keys(self):
        return {"split", "layer"}
