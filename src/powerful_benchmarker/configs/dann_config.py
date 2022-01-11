import torch
from pytorch_adapt.adapters import CDANNE, DANN, DANNE
from pytorch_adapt.containers import Misc, Models, Optimizers
from pytorch_adapt.hooks import (
    AFNHook,
    ATDOCHook,
    BNMHook,
    BSPHook,
    CLossHook,
    MCCHook,
    TargetDiversityHook,
    TargetEntropyHook,
)
from pytorch_adapt.layers import (
    AdaptiveFeatureNorm,
    L2PreservedDropout,
    MCCLoss,
    NLLLoss,
)
from pytorch_adapt.weighters import MeanWeighter

from powerful_benchmarker.utils import main_utils

from .base_config import BaseConfig
from .cdan_config import CDANConfig


class DANNConfig(BaseConfig):
    def get_adapter_kwargs(
        self, models, optimizers, before_training_starts, lr_multiplier, **kwargs
    ):
        models = Models(models)
        optimizers = Optimizers(optimizers, multipliers={"D": lr_multiplier})
        label_weight = self.optuna_trial.suggest_float("label_weight", 0, 1)
        domain_weight = self.optuna_trial.suggest_float("domain_weight", 0, 1)
        weighter = MeanWeighter(
            weights={
                "c_loss": label_weight,
                "src_domain_loss": domain_weight,
                "target_domain_loss": domain_weight,
            }
        )
        grl_weight = self.optuna_trial.suggest_float("grl_weight", 0.1, 10, log=True)
        hook_kwargs = {"weighter": weighter, "gradient_reversal_weight": grl_weight}

        return {
            "models": models,
            "optimizers": optimizers,
            "misc": None,
            "before_training_starts": before_training_starts,
            "hook_kwargs": hook_kwargs,
        }

    def get_new_adapter(self, *args, **kwargs):
        return DANN(**self.get_adapter_kwargs(*args, **kwargs))

    def save(self, folder):
        super().save(folder)
        main_utils.save_this_file(__file__, folder)


class DANNEConfig(DANNConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["detach_entropy_reducer"] = True
        return all_kwargs

    def get_new_adapter(self, *args, **kwargs):
        return DANNE(**self.get_adapter_kwargs(*args, **kwargs))


class DANNEUConfig(DANNEConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["detach_entropy_reducer"] = False
        return all_kwargs


class DANNTEConfig(DANNConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["post_g"] = [TargetEntropyHook()]
        entropy_weight = self.optuna_trial.suggest_float("entropy_weight", 0, 1)
        all_kwargs["hook_kwargs"]["weighter"].weights["entropy_loss"] = entropy_weight
        return all_kwargs


class DANNTEDConfig(DANNTEConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["post_g"] += [TargetDiversityHook()]
        weights = all_kwargs["hook_kwargs"]["weighter"].weights
        weights["diversity_loss"] = weights["entropy_loss"]
        return all_kwargs


class CDANNEUConfig(DANNConfig, CDANConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["misc"] = Misc(
            {"feature_combiner": all_kwargs["models"].pop("feature_combiner")}
        )
        all_kwargs["hook_kwargs"]["detach_entropy_reducer"] = False
        return all_kwargs

    def get_new_adapter(self, *args, **kwargs):
        return CDANNE(**self.get_adapter_kwargs(*args, **kwargs))


class BSPDANNConfig(DANNConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weight = self.optuna_trial.suggest_float("bsp_weight", 1e-6, 1, log=True)
        all_kwargs["hook_kwargs"]["weighter"].weights["bsp_loss"] = weight
        all_kwargs["hook_kwargs"]["post_g"] = [BSPHook()]
        return all_kwargs


class MCCDANNConfig(DANNConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weight = self.optuna_trial.suggest_float("mcc_weight", 0, 1)
        T = self.optuna_trial.suggest_float("T", 0.2, 5)
        loss_fn = MCCLoss(T=T)
        all_kwargs["hook_kwargs"]["weighter"].weights["mcc_loss"] = weight
        all_kwargs["hook_kwargs"]["post_g"] = [MCCHook(loss_fn=loss_fn)]
        return all_kwargs


class BNMDANNConfig(DANNConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weight = self.optuna_trial.suggest_float("bnm_weight", 0, 1)
        all_kwargs["hook_kwargs"]["weighter"].weights["bnm_loss"] = weight
        all_kwargs["hook_kwargs"]["post_g"] = [BNMHook()]
        return all_kwargs


class AFNDANNConfig(DANNConfig):
    def get_models(self, dataset, *args, **kwargs):
        models, framework = super().get_models(dataset, *args, **kwargs)
        fc = models["G"].fc
        if isinstance(fc, torch.nn.Sequential) and isinstance(fc[-1], torch.nn.Dropout):
            fc[-1] = L2PreservedDropout(p=fc[-1].p, inplace=fc[-1].inplace)
        return models, framework

    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weight = self.optuna_trial.suggest_float("afn_weight", 1e-6, 1, log=True)
        step_size = self.optuna_trial.suggest_float("step_size", 0, 2)
        loss_fn = AdaptiveFeatureNorm(step_size=step_size)
        all_kwargs["hook_kwargs"]["weighter"].weights["afn_loss"] = weight
        all_kwargs["hook_kwargs"]["post_g"] = [AFNHook(loss_fn=loss_fn)]
        return all_kwargs


class ATDOCDANNConfig(DANNConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weight = self.optuna_trial.suggest_float("atdoc_weight", 0, 1)
        k = self.optuna_trial.suggest_int("k", 5, 25, step=5)
        dataset_size = len(kwargs["datasets"]["target_train"])
        all_kwargs["hook_kwargs"]["weighter"].weights["pseudo_label_loss"] = weight
        hook = ATDOCHook(dataset_size, self.feature_size, self.num_classes, k=k)
        hook.labeler.to(torch.device("cuda"))
        all_kwargs["hook_kwargs"]["post_g"] = [hook]
        return all_kwargs


class DANNFL8Config(DANNConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["c_hook"] = CLossHook(
            loss_fn=NLLLoss(reduction="none")
        )
        return all_kwargs
