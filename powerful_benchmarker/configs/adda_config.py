from pytorch_adapt.adapters import ADDA
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.inference import adda_full_fn
from pytorch_adapt.weighters import MeanWeighter

from ..utils import main_utils
from .base_config import BaseConfig


class ADDAConfig(BaseConfig):
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
            multipliers={"D": lr_multiplier},
        )
        d_scale = self.optuna_trial.suggest_float("d_scale", 0, 1)
        g_scale = self.optuna_trial.suggest_float("g_scale", 0, 1)
        d_weighter = MeanWeighter(scale=d_scale)
        g_weighter = MeanWeighter(scale=g_scale)
        d_accuracy_threshold = self.optuna_trial.suggest_float(
            "d_accuracy_threshold", 0, 1
        )
        hook_kwargs = {
            "threshold": d_accuracy_threshold,
            "d_weighter": d_weighter,
            "g_weighter": g_weighter,
        }
        inference_fn = adda_full_fn if use_full_inference else None

        return {
            "models": models,
            "optimizers": optimizers,
            "misc": None,
            "before_training_starts": before_training_starts,
            "inference_fn": inference_fn,
            "hook_kwargs": hook_kwargs,
        }

    def get_new_adapter(self, *args, **kwargs):
        return ADDA(**self.get_adapter_kwargs(*args, **kwargs))

    def save(self, folder):
        super().save(folder)
        main_utils.save_this_file(__file__, folder)
