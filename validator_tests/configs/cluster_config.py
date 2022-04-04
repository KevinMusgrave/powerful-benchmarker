import torch.nn.functional as F
from pytorch_adapt.validators import ClusterValidator, KNNValidator
from sklearn.cluster import KMeans

from .base_config import use_labels_and_logits
from .knn_config import KNN


def kmeans_func(normalize, p):
    def fn(x, n_clusters):
        if normalize:
            x = F.normalize(x, dim=1, p=p)
        return KMeans(n_clusters=n_clusters).fit_predict(x.cpu().numpy())

    return fn


class DomainCluster(KNN):
    def create_validator(self, knn_func):
        return KNNValidator(
            key_map={
                self.src_split_name: "src_train",
                self.target_split_name: "target_train",
            },
            layer=self.layer,
            knn_func=knn_func,
            kmeans_func=kmeans_func(
                self.validator_args["normalize"], self.validator_args["p"]
            ),
            metric="AMI",
        )

    def set_k(self):
        pass

    def expected_keys(self):
        return {"p", "normalize", "layer", "split"}


class ClassCluster(KNN):
    def create_validator(self, knn_func):
        return ClusterValidator(
            key_map={
                self.src_split_name: "src_train",
                self.target_split_name: "target_train",
            },
            layer=self.layer,
            knn_func=knn_func,
            kmeans_func=kmeans_func(
                self.validator_args["normalize"], self.validator_args["p"]
            ),
            metric="AMI",
        )

    def score(self, x, exp_config, device):
        return use_labels_and_logits(
            x,
            device,
            self.validator,
            self.src_split_name,
            self.target_split_name,
            self.layer,
        )

    def set_k(self):
        pass

    def expected_keys(self):
        return {"p", "normalize", "layer", "split"}
