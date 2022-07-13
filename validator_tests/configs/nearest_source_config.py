from pytorch_adapt.validators import NearestSourceL2Validator, NearestSourceValidator

from .base_config import BaseConfig, get_split_and_layer


def get_score(x, layer, device, validator):
    src_features = get_split_and_layer(x, "src_val", layer, device)
    src_preds = get_split_and_layer(x, "src_val", "preds", device)
    src_labels = get_split_and_layer(x, "src_val", "labels", device)
    target_features = get_split_and_layer(x, "target_train", layer, device)
    return validator(
        src_val={
            layer: src_features,
            "preds": src_preds,
            "labels": src_labels,
        },
        target_train={layer: target_features},
    )


class NearestSource(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator_args["threshold"] = float(self.validator_args["threshold"])
        self.validator_args["weighted"] = bool(int(self.validator_args["weighted"]))
        self.layer = self.validator_args["layer"]
        self.validator = NearestSourceValidator(
            layer=self.layer,
            threshold=self.validator_args["threshold"],
            weighted=self.validator_args["weighted"],
        )

    def score(self, x, exp_config, device):
        return get_score(x, self.layer, device, self.validator)

    def expected_keys(self):
        return {"threshold", "weighted", "layer"}


class NearestSourceL2(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.layer = self.validator_args["layer"]
        self.validator = NearestSourceL2Validator(
            layer=self.layer,
        )

    def score(self, x, exp_config, device):
        return get_score(x, self.layer, device, self.validator)

    def expected_keys(self):
        return {"layer"}
