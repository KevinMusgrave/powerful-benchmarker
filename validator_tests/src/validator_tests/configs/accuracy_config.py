import torch.nn.functional as F
from pytorch_adapt.validators import AccuracyValidator

from powerful_benchmarker.utils.main_utils import num_classes

from .base_config import BaseConfig, get_from_hdf5


class Accuracy(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator = AccuracyValidator

    def actual_init(self, exp_config):
        self.validator = self.validator(
            key_map={self.split: "src_val"},
            torchmetric_kwargs={
                "average": self.validator_args["average"],
                "num_classes": num_classes(exp_config["dataset"]),
            },
        )

    def score(self, x, exp_config, device):
        if self.validator is AccuracyValidator:
            self.actual_init(exp_config)
        logits = get_from_hdf5(x, device, f"inference/{self.split}/logits")
        labels = get_from_hdf5(x, device, f"inference/{self.split}/labels")
        preds = F.softmax(logits, dim=1)
        return self.validator(**{self.split: {"preds": preds, "labels": labels}})
