import copy

from pytorch_adapt.adapters import MCD
from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.layers import (
    MCDLoss,
    MultipleModels,
    SlicedWasserstein,
    StochasticLinear,
)
from pytorch_adapt.utils.common_functions import reinit
from pytorch_adapt.weighters import MeanWeighter

from powerful_benchmarker.utils import main_utils

from .base_config import BaseConfig


class MCDConfig(BaseConfig):
    def get_models(self, dataset, *args, **kwargs):
        models, framework = super().get_models(dataset, *args, **kwargs)
        c1 = reinit(copy.deepcopy(models["C"]))
        models["C"] = MultipleModels(models["C"], c1)
        del models["D"]
        return models, framework

    def get_adapter_kwargs(
        self, models, optimizers, before_training_starts, lr_multiplier, **kwargs
    ):
        models = Models(models)
        optimizers = Optimizers(
            optimizers,
            multipliers={"C": lr_multiplier},
        )

        num_repeat = self.optuna_trial.suggest_int("num_repeat", 1, 10)

        label_weight = self.optuna_trial.suggest_float("label_weight", 0, 1)
        discrepancy_weight = self.optuna_trial.suggest_float("discrepancy_weight", 0, 1)

        x_weighter = MeanWeighter(scale=label_weight)
        y_weighter = MeanWeighter(
            weights={
                "c_loss0": label_weight,
                "c_loss1": label_weight,
                "discrepancy_loss": discrepancy_weight,
            }
        )
        z_weighter = MeanWeighter(scale=discrepancy_weight)
        hook_kwargs = {
            "repeat": num_repeat,
            "x_weighter": x_weighter,
            "y_weighter": y_weighter,
            "z_weighter": z_weighter,
        }

        return {
            "models": models,
            "optimizers": optimizers,
            "misc": None,
            "before_training_starts": before_training_starts,
            "hook_kwargs": hook_kwargs,
        }

    def get_new_adapter(self, *args, **kwargs):
        return MCD(**self.get_adapter_kwargs(*args, **kwargs))

    def save(self, folder):
        super().save(folder)
        main_utils.save_this_file(__file__, folder)


class STARConfig(MCDConfig):
    def get_models(self, dataset, *args, **kwargs):
        models, framework = super().get_models(dataset, *args, **kwargs)
        last_linear = models["C"].models[0].net[-1]
        in_features = last_linear.in_features
        out_features = last_linear.out_features
        new_linear = StochasticLinear(in_features, out_features)
        new_linear.weight_mean.data = last_linear.weight.data.t()
        new_linear.bias_mean.data = last_linear.bias.data.t()
        models["C"].models[0].net[-1] = new_linear
        models["C"].models[1] = models["C"].models[0]
        return models, framework


class SWDConfig(MCDConfig):
    def get_adapter_kwargs(self, *args, **kwargs):
        all_kwargs = super().get_adapter_kwargs(*args, **kwargs)
        m = 128
        all_kwargs["hook_kwargs"]["discrepancy_loss_fn"] = MCDLoss(
            dist_fn=SlicedWasserstein(m=m)
        )
        return all_kwargs
