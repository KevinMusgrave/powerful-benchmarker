from pytorch_adapt.layers.utils import get_kernel_scales
from pytorch_adapt.validators import MMDValidator
from pytorch_metric_learning.distances import LpDistance

from .base_config import BaseConfig, get_full_split_name, use_src_and_target


class MMD(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.layer = self.validator_args["layer"]
        self.src_split_name = get_full_split_name("src", self.split)
        self.target_split_name = get_full_split_name("target", self.split)

        exponent = int(self.validator_args["exponent"])
        num_kernels = (exponent * 2) + 1
        kernel_scales = get_kernel_scales(
            low=-exponent, high=exponent, num_kernels=num_kernels
        )
        normalize = bool(int(self.validator_args["normalize"]))
        dist_func = LpDistance(normalize_embeddings=normalize, p=2, power=2)

        self.validator = MMDValidator(
            key_map={
                self.src_split_name: "src_train",
                self.target_split_name: "target_train",
            },
            layer=self.validator_args["layer"],
            num_samples=4096,
            num_trials=1000,
            mmd_kwargs={"kernel_scales": kernel_scales, "dist_func": dist_func},
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
