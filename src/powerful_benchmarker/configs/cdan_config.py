from pytorch_adapt.adapters import CDAN, CDANE
from pytorch_adapt.containers import Misc, Models, Optimizers
from pytorch_adapt.layers import RandomizedDotProduct
from pytorch_adapt.weighters import MeanWeighter

from ..utils import main_utils
from .base_config import BaseConfig


class CDANConfig(BaseConfig):
    def get_models(self, dataset, *args, **kwargs):
        num_classes = main_utils.num_classes(dataset)
        models, framework = super().get_models(dataset, *args, **kwargs)
        models["feature_combiner"] = RandomizedDotProduct(
            [self.feature_size, num_classes], self.feature_size
        )
        return models, framework

    def get_adapter_kwargs(
        self, models, optimizers, before_training_starts, lr_multiplier, **kwargs
    ):
        feature_combiner = models.pop("feature_combiner")
        models = Models(models)
        optimizers = Optimizers(optimizers, multipliers={"D": lr_multiplier})
        misc = Misc({"feature_combiner": feature_combiner})

        d_weight = self.optuna_trial.suggest_float("d_weight", 0, 1)
        g_weight = self.optuna_trial.suggest_float("g_weight", 0, 1)
        label_weight = self.optuna_trial.suggest_float("label_weight", 0, 1)

        d_weighter = MeanWeighter(scale=d_weight)
        g_weighter = MeanWeighter(
            weights={
                "g_src_domain_loss": g_weight,
                "g_target_domain_loss": g_weight,
                "c_loss": label_weight,
            },
        )
        hook_kwargs = {
            "d_weighter": d_weighter,
            "g_weighter": g_weighter,
        }
        return {
            "models": models,
            "optimizers": optimizers,
            "misc": misc,
            "before_training_starts": before_training_starts,
            "hook_kwargs": hook_kwargs,
        }

    def get_new_adapter(self, *args, **kwargs):
        return CDAN(**self.get_adapter_kwargs(*args, **kwargs))

    def save(self, folder):
        super().save(folder)
        main_utils.save_this_file(__file__, folder)


class CDANEConfig(CDANConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["detach_entropy_reducer"] = True
        return all_kwargs

    def get_new_adapter(self, *args, **kwargs):
        return CDANE(**self.get_adapter_kwargs(*args, **kwargs))


class CDANEUConfig(CDANEConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["detach_entropy_reducer"] = False
        return all_kwargs
