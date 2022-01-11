import torch
from pytorch_adapt.adapters import Classifier, Finetuner
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.hooks import (
    AFNHook,
    ATDOCHook,
    BNMHook,
    BSPHook,
    ISTLossHook,
    MCCHook,
    TargetDiversityHook,
    TargetEntropyHook,
)
from pytorch_adapt.layers import AdaptiveFeatureNorm, L2PreservedDropout, MCCLoss
from pytorch_adapt.weighters import MeanWeighter

from powerful_benchmarker.utils import main_utils

from .base_config import BaseConfig


class PretrainerConfig(BaseConfig):
    def get_adapter_kwargs(
        self, models, optimizers, before_training_starts, lr_multiplier, **kwargs
    ):
        del models["D"]
        models = Models(models)
        optimizers = Optimizers(optimizers, multipliers={"C": lr_multiplier})
        return {
            "models": models,
            "optimizers": optimizers,
            "misc": None,
            "before_training_starts": before_training_starts,
            "hook_kwargs": {},
        }

    def get_new_adapter(self, *args, **kwargs):
        return Classifier(**self.get_adapter_kwargs(*args, **kwargs))

    def save(self, folder):
        super().save(folder)
        main_utils.save_this_file(__file__, folder)


class FinetunerConfig(PretrainerConfig):
    def get_new_adapter(self, *args, **kwargs):
        return Finetuner(**self.get_adapter_kwargs(*args, **kwargs))


class ClassifierConfig(PretrainerConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        label_weight = self.optuna_trial.suggest_float("label_weight", 0, 1)
        weighter = MeanWeighter(
            weights={
                "c_loss": label_weight,
            }
        )
        all_kwargs["hook_kwargs"]["weighter"] = weighter
        return all_kwargs


class BSPConfig(ClassifierConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weight = self.optuna_trial.suggest_float("bsp_weight", 1e-6, 1, log=True)
        all_kwargs["hook_kwargs"]["weighter"].weights["bsp_loss"] = weight
        all_kwargs["hook_kwargs"]["post"] = [BSPHook()]
        return all_kwargs


class MCCConfig(ClassifierConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weight = self.optuna_trial.suggest_float("mcc_weight", 0, 1)
        T = self.optuna_trial.suggest_float("T", 0.2, 5)
        loss_fn = MCCLoss(T=T)
        all_kwargs["hook_kwargs"]["weighter"].weights["mcc_loss"] = weight
        all_kwargs["hook_kwargs"]["post"] = [MCCHook(loss_fn=loss_fn)]
        return all_kwargs


class BNMConfig(ClassifierConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weight = self.optuna_trial.suggest_float("bnm_weight", 0, 1)
        all_kwargs["hook_kwargs"]["weighter"].weights["bnm_loss"] = weight
        all_kwargs["hook_kwargs"]["post"] = [BNMHook()]
        return all_kwargs


class AFNConfig(ClassifierConfig):
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
        all_kwargs["hook_kwargs"]["post"] = [AFNHook(loss_fn=loss_fn)]
        return all_kwargs


class ATDOCConfig(ClassifierConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weight = self.optuna_trial.suggest_float("atdoc_weight", 0, 1)
        k = self.optuna_trial.suggest_int("k", 5, 25, step=5)
        dataset_size = len(kwargs["datasets"]["target_train"])
        all_kwargs["hook_kwargs"]["weighter"].weights["pseudo_label_loss"] = weight
        hook = ATDOCHook(dataset_size, self.feature_size, self.num_classes, k=k)
        hook.labeler.to(torch.device("cuda"))
        all_kwargs["hook_kwargs"]["post"] = [hook]
        return all_kwargs


class TEConfig(ClassifierConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weight = self.optuna_trial.suggest_float("entropy_weight", 0, 1)
        all_kwargs["hook_kwargs"]["weighter"].weights["entropy_loss"] = weight
        all_kwargs["hook_kwargs"]["post"] = [TargetEntropyHook()]
        return all_kwargs


class TEDConfig(TEConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        all_kwargs["hook_kwargs"]["post"] += [TargetDiversityHook()]
        weights = all_kwargs["hook_kwargs"]["weighter"].weights
        weights["diversity_loss"] = weights["entropy_loss"]
        return all_kwargs


class ITLConfig(TEDConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        weight = self.optuna_trial.suggest_float("ist_weight", 0, 1)
        all_kwargs["hook_kwargs"]["weighter"].weights["ist_loss"] = weight
        all_kwargs["hook_kwargs"]["post"] += [ISTLossHook()]
        return all_kwargs
