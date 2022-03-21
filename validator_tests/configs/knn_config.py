from pytorch_adapt.validators.knn_validator import KNNValidator
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN

from .base_config import BaseConfig, get_full_split_name, use_src_and_target


class KNN(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator_args["k"] = int(self.validator_args["k"])
        self.validator_args["p"] = float(self.validator_args["p"])
        self.validator_args["normalize"] = bool(int(self.validator_args["normalize"]))
        self.layer = self.validator_args["layer"]
        self.src_split_name = get_full_split_name("src", self.split)
        self.target_split_name = get_full_split_name("target", self.split)

        knn_func = CustomKNN(
            LpDistance(
                normalize_embeddings=self.validator_args["normalize"],
                p=self.validator_args["p"],
            ),
            batch_size=512,
        )

        self.validator = KNNValidator(
            key_map={
                self.src_split_name: "src_train",
                self.target_split_name: "target_train",
            },
            layer=self.validator_args["layer"],
            knn_func=knn_func,
            k=self.validator_args["k"],
            metric="mean_average_precision",
        )

    def score(self, x, exp_config, device):
        return use_src_and_target(
            x,
            device,
            self.validator,
            self.src_split_name,
            self.target_split_name,
            self.layer,
        )
