from pytorch_adapt.adapters import GAN, GANE, DomainConfusion
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.hooks import CLossHook, TargetDiversityHook, TargetEntropyHook
from pytorch_adapt.layers import NLLLoss
from pytorch_adapt.models import Discriminator
from pytorch_adapt.weighters import MeanWeighter

from ..utils import main_utils
from .base_config import BaseConfig


class GANConfig(BaseConfig):
    def get_adapter_kwargs(
        self, models, optimizers, before_training_starts, lr_multiplier, **kwargs
    ):
        models = Models(models)
        optimizers = Optimizers(optimizers, multipliers={"D": lr_multiplier})
        label_weight = self.optuna_trial.suggest_float("label_weight", 0, 1)
        d_weight = self.optuna_trial.suggest_float("d_weight", 0, 1)
        g_weight = self.optuna_trial.suggest_float("g_weight", 0, 1)

        d_loss_weighter = MeanWeighter(scale=d_weight)
        g_loss_weighter = MeanWeighter(
            weights={
                "c_loss": label_weight,
                "g_src_domain_loss": g_weight,
                "g_target_domain_loss": g_weight,
            },
        )

        hook_kwargs = {"d_weighter": d_loss_weighter, "g_weighter": g_loss_weighter}

        return {
            "models": models,
            "optimizers": optimizers,
            "misc": None,
            "before_training_starts": before_training_starts,
            "hook_kwargs": hook_kwargs,
        }

    def get_new_adapter(self, *args, **kwargs):
        return GAN(**self.get_adapter_kwargs(*args, **kwargs))

    def save(self, folder):
        super().save(folder)
        main_utils.save_this_file(__file__, folder)


class DomainConfusionConfig(GANConfig):
    def get_models(self, dataset, *args, **kwargs):
        models, framework = super().get_models(dataset, *args, **kwargs)
        h = models["D"].h
        models["D"] = Discriminator(in_size=self.feature_size, h=h, out_size=2)
        return models, framework

    def get_new_adapter(self, *args, **kwargs):
        return DomainConfusion(**self.get_adapter_kwargs(*args, **kwargs))


class GANEConfig(GANConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["detach_entropy_reducer"] = True
        return all_kwargs

    def get_new_adapter(self, *args, **kwargs):
        return GANE(**self.get_adapter_kwargs(*args, **kwargs))


class GANEUConfig(GANEConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["detach_entropy_reducer"] = False
        return all_kwargs


class GANMinEntConfig(GANConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["post_g"] = [TargetEntropyHook()]
        entropy_weight = self.optuna_trial.suggest_float("entropy_weight", 0, 1)
        all_kwargs["hook_kwargs"]["g_weighter"].weights["entropy_loss"] = entropy_weight
        return all_kwargs


class GANIMConfig(GANMinEntConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["post_g"] += [TargetDiversityHook()]
        weights = all_kwargs["hook_kwargs"]["g_weighter"].weights
        weights["diversity_loss"] = weights["entropy_loss"]
        return all_kwargs


class GANFL8Config(GANConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["c_hook"] = CLossHook(
            loss_fn=NLLLoss(reduction="none")
        )
        return all_kwargs
