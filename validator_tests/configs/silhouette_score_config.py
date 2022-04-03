from pytorch_adapt.validators import CHScoreValidator, SilhouetteScoreValidator

from .base_config import BaseConfig, get_split_and_layer


class SilhouetteScore(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.layer = self.validator_args["layer"]
        self.validator_args["with_src"] = bool(int(self.validator_args["with_src"]))
        self.validator_args["pca_size"] = int(self.validator_args["pca_size"])
        self.validator = SilhouetteScoreValidator(
            layer=self.layer,
            with_src=self.validator_args["with_src"],
            pca_size=self.validator_args["pca_size"],
        )

    def score(self, x, exp_config, device):
        kwargs = {}
        if self.validator_args["with_src"]:
            kwargs["src_train"] = {
                self.layer: get_split_and_layer(x, "src_train", self.layer, device)
            }
        target_features = get_split_and_layer(x, "target_train", self.layer, device)
        target_logits = get_split_and_layer(x, "target_train", "logits", device)
        kwargs["target_train"] = {self.layer: target_features, "logits": target_logits}
        return self.validator(**kwargs)

    def expected_keys(self):
        return {"layer", "with_src", "pca_size"}


class CHScore(SilhouetteScore):
    def __init__(self, config):
        super().__init__(config)
        self.validator = CHScoreValidator(
            layer=self.layer,
            with_src=self.validator_args["with_src"],
            pca_size=self.validator_args["pca_size"],
        )
