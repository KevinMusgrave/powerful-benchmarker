from pytorch_adapt.validators import KNNValidator, TargetKNNValidator
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN

from .base_config import BaseConfig, get_full_split_name, use_src_and_target


class KNN(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator_args["k"] = int(self.validator_args["k"])
        self.validator_args["p"] = float(self.validator_args["p"])
        self.validator_args["normalize"] = bool(int(self.validator_args["normalize"]))
        self.set_layer()
        self.src_split_name = get_full_split_name("src", self.split)
        self.target_split_name = get_full_split_name("target", self.split)

        knn_func = CustomKNN(
            LpDistance(
                normalize_embeddings=self.validator_args["normalize"],
                p=self.validator_args["p"],
            ),
            batch_size=512,
        )

        self.validator = self.create_validator(knn_func)

    def score(self, x, exp_config, device):
        return use_src_and_target(
            x,
            device,
            self.validator,
            self.src_split_name,
            self.target_split_name,
            self.layer,
        )

    def create_validator(self, knn_func):
        return KNNValidator(
            key_map={
                self.src_split_name: "src_train",
                self.target_split_name: "target_train",
            },
            layer=self.layer,
            knn_func=knn_func,
            k=self.validator_args["k"],
            metric="mean_average_precision",
        )

    def set_layer(self):
        self.layer = self.validator_args["layer"]

    def expected_keys(self):
        return {"k", "p", "normalize", "layer", "split"}


class TargetKNN(KNN):
    def create_validator(self, knn_func):
        self.validator_args["T_in_ref"] = bool(int(self.validator_args["T_in_ref"]))
        return TargetKNNValidator(
            key_map={
                self.src_split_name: "src_train",
                self.target_split_name: "target_train",
            },
            layer=self.layer,
            knn_func=knn_func,
            k=self.validator_args["k"],
            metric="mean_average_precision",
            add_target_to_ref=self.validator_args["T_in_ref"],
        )

    def set_layer(self):
        self.layer = "features"

    def expected_keys(self):
        return {"k", "p", "normalize", "T_in_ref", "split"}
