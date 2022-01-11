from pytorch_adapt.adapters import VADA
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.layers import VATLoss
from pytorch_adapt.weighters import MeanWeighter

from powerful_benchmarker.utils import main_utils

from .base_config import BaseConfig


class VADAConfig(BaseConfig):
    def get_adapter_kwargs(
        self, models, optimizers, before_training_starts, lr_multiplier, **kwargs
    ):
        models = Models(models)
        optimizers = Optimizers(optimizers, multipliers={"D": lr_multiplier})
        d_weight = self.optuna_trial.suggest_float("d_weight", 0, 1)
        g_weight = self.optuna_trial.suggest_float("g_weight", 0, 1)
        src_weight = self.optuna_trial.suggest_float("src_weight", 0, 1)
        target_weight = self.optuna_trial.suggest_float("target_weight", 0, 1)
        d_loss_weighter = MeanWeighter(scale=d_weight)
        g_loss_weighter = MeanWeighter(
            weights={
                "src_vat_loss": src_weight,
                "target_vat_loss": target_weight,
                "entropy_loss": target_weight,
                "g_src_domain_loss": g_weight,
                "g_target_domain_loss": g_weight,
            }
        )

        vat_loss_fn = VATLoss(epsilon=2)

        hook_kwargs = {
            "d_weighter": d_loss_weighter,
            "g_weighter": g_loss_weighter,
            "vat_loss_fn": vat_loss_fn,
        }

        return {
            "models": models,
            "optimizers": optimizers,
            "misc": None,
            "before_training_starts": before_training_starts,
            "hook_kwargs": hook_kwargs,
        }

    def get_new_adapter(self, *args, **kwargs):
        return VADA(**self.get_adapter_kwargs(*args, **kwargs))

    def save(self, folder):
        super().save(folder)
        main_utils.save_this_file(__file__, folder)
