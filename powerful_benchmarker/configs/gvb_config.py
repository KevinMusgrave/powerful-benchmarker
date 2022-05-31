import torch
from pytorch_adapt.adapters import GVB, GVBE
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.inference import gvb_full_fn
from pytorch_adapt.layers import ModelWithBridge
from pytorch_adapt.models import Discriminator
from pytorch_adapt.weighters import MeanWeighter

from ..utils import main_utils
from .base_config import BaseConfig


class Bridge(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc = torch.nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.fc(x).squeeze(1)


class GVBConfig(BaseConfig):
    def get_models(self, *args, **kwargs):
        models, framework = super().get_models(*args, **kwargs)
        num_classes = kwargs["num_classes"]
        h = models["D"].h
        models["D"] = Discriminator(in_size=num_classes, h=h)
        models["D"] = ModelWithBridge(models["D"], Bridge(num_classes, 1))
        models["C"] = ModelWithBridge(
            models["C"], Bridge(self.feature_size, num_classes)
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
        models = Models(models)
        optimizers = Optimizers(
            optimizers,
            multipliers={
                "C": lr_multiplier,
                "D": lr_multiplier,
            },
        )
        domain_weight = self.optuna_trial.suggest_float("domain_weight", 0, 1)
        bridge_G_weight = self.optuna_trial.suggest_float("bridge_G_weight", 0, 1)
        bridge_D_weight = self.optuna_trial.suggest_float("bridge_D_weight", 0, 1)
        weighter = MeanWeighter(
            weights={
                "src_domain_loss": domain_weight,
                "target_domain_loss": domain_weight,
                "g_src_bridge_loss": bridge_G_weight,
                "d_src_bridge_loss": bridge_D_weight,
                "g_target_bridge_loss": bridge_G_weight,
                "d_target_bridge_loss": bridge_D_weight,
            }
        )

        grl_weight = self.optuna_trial.suggest_float("grl_weight", 0.1, 10, log=True)
        hook_kwargs = {"weighter": weighter, "gradient_reversal_weight": grl_weight}
        inference_fn = gvb_full_fn if use_full_inference else None

        return {
            "models": models,
            "optimizers": optimizers,
            "misc": None,
            "before_training_starts": before_training_starts,
            "inference_fn": inference_fn,
            "hook_kwargs": hook_kwargs,
        }

    def get_new_adapter(self, *args, **kwargs):
        return GVB(**self.get_adapter_kwargs(*args, **kwargs))

    def save(self, folder):
        super().save(folder)
        main_utils.save_this_file(__file__, folder)


class GVBEConfig(GVBConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["detach_entropy_reducer"] = True
        return all_kwargs

    def get_new_adapter(self, *args, **kwargs):
        return GVBE(**self.get_adapter_kwargs(*args, **kwargs))


class GVBEUConfig(GVBEConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["detach_entropy_reducer"] = False
        return all_kwargs
