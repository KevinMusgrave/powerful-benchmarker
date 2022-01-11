from pytorch_adapt.adapters import Aligner
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.hooks import JointAlignerHook
from pytorch_adapt.layers import CORALLoss, MMDLoss
from pytorch_adapt.layers.utils import get_kernel_scales
from pytorch_adapt.weighters import MeanWeighter

from powerful_benchmarker.utils import main_utils

from .base_config import BaseConfig


class AlignerConfig(BaseConfig):
    def get_models(self, *args, **kwargs):
        models, framework = super().get_models(*args, **kwargs)
        del models["D"]
        return models, framework

    def get_adapter_kwargs(
        self, models, optimizers, before_training_starts, lr_multiplier, **kwargs
    ):
        models = Models(models)
        optimizers = Optimizers(optimizers, multipliers={"C": lr_multiplier})

        confusion_weight = self.optuna_trial.suggest_float("confusion_weight", 0, 1)
        label_weight = self.optuna_trial.suggest_float("label_weight", 0, 1)
        weighter = MeanWeighter(
            weights={
                "features_confusion_loss": confusion_weight,
                "logits_confusion_loss": confusion_weight,
                "c_loss": label_weight,
            }
        )
        hook_kwargs = {"weighter": weighter}
        return {
            "models": models,
            "optimizers": optimizers,
            "misc": None,
            "before_training_starts": before_training_starts,
            "hook_kwargs": hook_kwargs,
        }

    def get_new_adapter(self, *args, **kwargs):
        return Aligner(**self.get_adapter_kwargs(*args, **kwargs))

    def save(self, folder):
        super().save(folder)
        main_utils.save_this_file(__file__, folder)


class CORALConfig(AlignerConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"].update({"loss_fn": CORALLoss(), "softmax": False})
        return all_kwargs


class MMDConfig(AlignerConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        exponent = self.optuna_trial.suggest_int("exponent", 1, 8)
        num_kernels = (exponent * 2) + 1
        kernel_scales = get_kernel_scales(
            low=-exponent, high=exponent, num_kernels=num_kernels
        )
        all_kwargs["hook_kwargs"].update(
            {
                "loss_fn": MMDLoss(kernel_scales=kernel_scales),
                "softmax": True,
            }
        )
        return all_kwargs


class JMMDConfig(MMDConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weighter = MeanWeighter(
            weights={
                "joint_confusion_loss": self.optuna_trial.params["confusion_weight"],
                "c_loss": self.optuna_trial.params["label_weight"],
            }
        )
        aligner_hook = JointAlignerHook(
            loss_fn=all_kwargs["hook_kwargs"].pop("loss_fn")
        )
        all_kwargs["hook_kwargs"].update(
            {"weighter": weighter, "aligner_hook": aligner_hook}
        )
        return all_kwargs
