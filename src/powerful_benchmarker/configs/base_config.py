import torch
from pytorch_adapt.frameworks.ignite import IgnitePredsAsFeatures
from pytorch_adapt.layers import DoNothingOptimizer
from pytorch_adapt.models import Discriminator
from pytorch_adapt.models import pretrained as pretrained_module
from pytorch_adapt.utils.common_functions import get_lr

from powerful_benchmarker.utils import main_utils

from ..utils import main_utils


class BaseConfig:
    def __init__(self, optuna_trial):
        self.optuna_trial = optuna_trial

    def get_optimizers(self, pretrain_on_src, optimizer_name, pretrain_lr):
        if pretrain_on_src:
            lr = pretrain_lr
        else:
            lr = self.optuna_trial.suggest_float("lr", 1e-5, 0.1, log=True)
        if optimizer_name == "SGD":
            return (
                torch.optim.SGD,
                {"lr": lr, "momentum": 0.9, "weight_decay": 1e-4},
            )
        elif optimizer_name == "Adam":
            return (torch.optim.Adam, {"lr": lr, "weight_decay": 1e-4})
        else:
            raise TypeError

    def get_before_training_starts_hook(self, optimizer_name):
        def before_training_starts(cls):
            def func(framework):
                _, max_iters = framework.get_training_length()
                print("max_iters", max_iters)
                for k, v in cls.optimizers.items():
                    if not isinstance(v, DoNothingOptimizer):
                        cls.lr_schedulers[k] = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer=v,
                            max_lr=get_lr(v),
                            total_steps=max_iters,
                            pct_start=0.05,
                            anneal_strategy="cos",
                            div_factor=100,
                            final_div_factor=float("inf"),
                        )
                cls.lr_schedulers.scheduler_types = {
                    "per_step": list(cls.lr_schedulers.keys()),
                    "per_epoch": [],
                }
                cls.before_training_starts_default(framework)

            return func

        return before_training_starts

    def get_models(
        self,
        dataset,
        src_domains,
        start_with_pretrained,
        pretrain_on_src,
        num_classes,
        feature_layer,
    ):
        assert len(src_domains) == 1
        self.num_classes = num_classes

        kwargs = {"pretrained": start_with_pretrained}
        G = getattr(pretrained_module, f"{dataset}G")(**kwargs)

        if dataset != "mnist":
            kwargs["domain"] = src_domains[0]
        C = getattr(pretrained_module, f"{dataset}C")(**kwargs)

        models = {"G": G, "C": C}
        models, self.feature_size, framework = self.set_feature_layer(
            models, dataset, pretrain_on_src, feature_layer
        )
        models["D"] = Discriminator(in_size=self.feature_size, h=2048)
        return models, framework

    def set_feature_layer(self, models, dataset, pretrain_on_src, feature_layer):
        fc_out_feature_size = {
            "mnist": 1200,
            "domainnet": 2048,
            "domainnet126": 2048,
            "office31": 2048,
            "officehome": 2048,
        }[dataset]
        framework = None
        if pretrain_on_src or feature_layer == 0:
            return models, fc_out_feature_size, framework
        if feature_layer in [7, 8]:
            models["G"].fc = models["C"].net
            models["C"].net = torch.nn.Identity()
            if feature_layer == 8:
                models["G"].fc = torch.nn.Sequential(
                    *models["G"].fc, torch.nn.Softmax(dim=1)
                )
                framework = IgnitePredsAsFeatures
        else:
            models["G"].fc = models["C"].net[:feature_layer]
            models["C"].net = models["C"].net[feature_layer:]
        if feature_layer in [3, 6]:
            f_idx = -3
        elif feature_layer in [2, 5, 8]:
            f_idx = -2
        else:
            f_idx = -1
        return models, models["G"].fc[f_idx].out_features, framework

    def save(self, folder):
        main_utils.save_this_file(__file__, folder)
