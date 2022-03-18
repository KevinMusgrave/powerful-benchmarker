from pytorch_adapt.adapters import SymNets
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.inference import symnets_full_fn
from pytorch_adapt.weighters import MeanWeighter

from ..utils import main_utils
from .mcd_config import MCDConfig


class SymNetsConfig(MCDConfig):
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
            multipliers={"C": lr_multiplier},
        )

        domain_weight = self.optuna_trial.suggest_float("domain_weight", 0, 1)
        category_weight = self.optuna_trial.suggest_float("category_weight", 0, 1)
        confusion_weight = self.optuna_trial.suggest_float("confusion_weight", 0, 1)
        entropy_weight = self.optuna_trial.suggest_float("entropy_weight", 0, 1)

        c_weighter = MeanWeighter(
            weights={
                "c_symnets_src_domain_loss_0": domain_weight,
                "c_symnets_target_domain_loss_1": domain_weight,
            }
        )
        g_weighter = MeanWeighter(
            weights={
                "symnets_category_loss": category_weight,
                "g_symnets_target_domain_loss_0": confusion_weight,
                "g_symnets_target_domain_loss_1": confusion_weight,
                "symnets_entropy_loss": entropy_weight,
            }
        )

        hook_kwargs = {"c_weighter": c_weighter, "g_weighter": g_weighter}

        inference_fn = symnets_full_fn if use_full_inference else None

        return {
            "models": models,
            "optimizers": optimizers,
            "misc": None,
            "before_training_starts": before_training_starts,
            "inference_fn": inference_fn,
            "hook_kwargs": hook_kwargs,
        }

    def get_new_adapter(self, *args, **kwargs):
        return SymNets(**self.get_adapter_kwargs(*args, **kwargs))

    def save(self, folder):
        super().save(folder)
        main_utils.save_this_file(__file__, folder)
