import torch
from pytorch_adapt.adapters import RTN
from pytorch_adapt.containers import Misc, Models, Optimizers
from pytorch_adapt.datasets import utils as dataset_utils
from pytorch_adapt.inference import rtn_full_fn
from pytorch_adapt.layers import MMDLoss, PlusResidual, RandomizedDotProduct
from pytorch_adapt.layers.utils import get_kernel_scales
from pytorch_adapt.weighters import MeanWeighter

from ..utils import main_utils
from .base_config import BaseConfig


class RTNConfig(BaseConfig):
    def get_models(self, dataset, *args, **kwargs):
        models, framework = super().get_models(dataset, *args, **kwargs)
        num_classes = dataset_utils.num_classes(dataset)
        models["residual_model"] = PlusResidual(
            torch.nn.Sequential(
                torch.nn.Linear(num_classes, num_classes),
                torch.nn.ReLU(),
                torch.nn.Linear(num_classes, num_classes),
            )
        )
        del models["D"]
        models["feature_combiner"] = RandomizedDotProduct(
            [self.feature_size, num_classes], self.feature_size
        )
        return models, framework

    def get_adapter_kwargs(
        self,
        models,
        optimizers,
        before_training_starts,
        lr_multiplier,
        use_full_inference,
        **kwargs
    ):
        feature_combiner = models.pop("feature_combiner")
        models = Models(models)
        optimizers = Optimizers(
            optimizers, multipliers={"residual_model": lr_multiplier}
        )
        misc = Misc({"feature_combiner": feature_combiner})

        label_weight = self.optuna_trial.suggest_float("label_weight", 0, 1)

        confusion_weight = self.optuna_trial.suggest_float("confusion_weight", 0, 1)
        entropy_weight = self.optuna_trial.suggest_float("entropy_weight", 0, 1)
        weighter = MeanWeighter(
            weights={
                "c_loss": label_weight,
                "features_confusion_loss": confusion_weight,
                "entropy_loss": entropy_weight,
            }
        )

        exponent = self.optuna_trial.suggest_int("exponent", 1, 8)
        num_kernels = (exponent * 2) + 1
        kernel_scales = get_kernel_scales(
            low=-exponent, high=exponent, num_kernels=num_kernels
        )
        hook_kwargs = {
            "weighter": weighter,
            "aligner_loss_fn": MMDLoss(kernel_scales=kernel_scales),
        }

        inference_fn = rtn_full_fn if use_full_inference else None

        return {
            "models": models,
            "optimizers": optimizers,
            "misc": misc,
            "before_training_starts": before_training_starts,
            "inference_fn": inference_fn,
            "hook_kwargs": hook_kwargs,
        }

    def get_new_adapter(self, *args, **kwargs):
        return RTN(**self.get_adapter_kwargs(*args, **kwargs))

    def save(self, folder):
        super().save(folder)
        main_utils.save_this_file(__file__, folder)
