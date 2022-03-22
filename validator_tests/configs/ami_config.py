import torch.nn.functional as F
from pytorch_adapt.validators import ClusterValidator
from sklearn.cluster import KMeans

from .base_config import BaseConfig, get_full_split_name, use_src_and_target


# pass in dummy to avoid initializing FaissKNN
def knn_func():
    pass


def kmeans_func(normalize):
    def fn(x, n_clusters):
        if normalize:
            x = F.normalize(x, dim=1, p=2)
        return KMeans(n_clusters=n_clusters).fit_predict(x.cpu().numpy())

    return fn


class AMI(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator_args["normalize"] = bool(int(self.validator_args["normalize"]))
        self.layer = self.validator_args["layer"]
        self.src_split_name = get_full_split_name("src", self.split)
        self.target_split_name = get_full_split_name("target", self.split)

        self.validator = ClusterValidator(
            key_map={
                self.src_split_name: "src_train",
                self.target_split_name: "target_train",
            },
            layer=self.validator_args["layer"],
            knn_func=knn_func,
            kmeans_func=kmeans_func(self.validator_args["normalize"]),
            metric="AMI",
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
