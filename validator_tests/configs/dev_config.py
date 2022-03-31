from pytorch_adapt.validators import DeepEmbeddedValidator

from .base_config import BaseConfig, get_split_and_layer


class DEV(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.layer = self.validator_args["layer"]
        if self.validator_args["normalization"] == "None":
            self.validator_args["normalization"] = None
        self.validator = DeepEmbeddedValidator(
            temp_folder=None,
            batch_size=256,
            layer=self.layer,
            normalization=self.validator_args["normalization"],
        )

    def score(self, x, exp_config, device):
        src_train = get_split_and_layer(x, "src_train", self.layer, device)
        src_val = get_split_and_layer(x, "src_val", self.layer, device)
        src_val_labels = get_split_and_layer(x, "src_val", "labels", device)
        src_val_logits = get_split_and_layer(x, "src_val", "logits", device)
        target_train = get_split_and_layer(x, "target_train", self.layer, device)
        return self.validator(
            src_train={self.layer: src_train},
            src_val={
                self.layer: src_val,
                "labels": src_val_labels,
                "logits": src_val_logits,
            },
            target_train={self.layer: target_train},
        )

    def expected_keys(self):
        return {"layer", "min_var"}
