import torch.nn.functional as F
from pytorch_adapt.validators import ClassClusterValidator, KNNValidator
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

from .base_config import BaseConfig, get_full_split_name, use_labels_and_logits
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


def feat_normalizer_fn(normalize, p):
    def fn(x):
        if normalize:
            return F.normalize(x, dim=1, p=p)
        return x

    return fn


class ClassAMI(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.layer = self.validator_args["layer"]
        self.validator_args["p"] = float(self.validator_args["p"])
        self.validator_args["with_src"] = bool(int(self.validator_args["with_src"]))
        self.validator_args["normalize"] = bool(int(self.validator_args["normalize"]))
        self.src_split_name = get_full_split_name("src", self.split)
        self.target_split_name = get_full_split_name("target", self.split)
        self.create_validator()

    def create_validator(self):
        score_fn, score_fn_type = self.get_score_fn()
        self.validator = ClassClusterValidator(
            key_map={
                self.src_split_name: "src_train",
                self.target_split_name: "target_train",
            },
            layer=self.layer,
            score_fn=score_fn,
            score_fn_type=score_fn_type,
            with_src=self.validator_args["with_src"],
            pca_size=None,
            centroid_init=self.get_centroid_init(),
            feat_normalizer=feat_normalizer_fn(
                self.validator_args["normalize"], self.validator_args["p"]
            ),
        )

    def get_score_fn(self):
        return adjusted_mutual_info_score, "labels"

    def get_centroid_init(self):
        return None

    def score(self, x, exp_config, device):
        return use_labels_and_logits(
            x,
            device,
            self.validator,
            self.src_split_name,
            self.target_split_name,
            self.layer,
        )

    def expected_keys(self):
        return {"p", "with_src", "normalize", "layer", "split"}


class ClassAMICentroidInit(ClassAMI):
    def get_centroid_init(self):
        return "label_centers"


class ClassSS(ClassAMI):
    def get_score_fn(self):
        return silhouette_score, "features"


class ClassSSCentroidInit(ClassSS):
    def get_centroid_init(self):
        return "label_centers"
