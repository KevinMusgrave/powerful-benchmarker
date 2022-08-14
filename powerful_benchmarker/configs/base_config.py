import torch
from pytorch_adapt.frameworks.ignite import (
    Ignite,
    IgniteMultiLabelClassification,
    IgnitePredsAsFeatures,
)
from pytorch_adapt.layers import DoNothingOptimizer
from pytorch_adapt.models import Discriminator
from pytorch_adapt.models import pretrained as pretrained_module
from pytorch_adapt.utils.common_functions import get_lr

from ..utils import main_utils


class BaseConfig:
    def __init__(self, optuna_trial):
        self.optuna_trial = optuna_trial

    def get_optimizers(self, pretrain_on_src, optimizer, pretrain_lr):
        if pretrain_on_src:
            lr = pretrain_lr
        else:
            lr = self.optuna_trial.suggest_float("lr", 1e-5, 0.1, log=True)
        if optimizer == "SGD":
            return (
                torch.optim.SGD,
                {"lr": lr, "momentum": 0.9, "weight_decay": 1e-4},
            )
        elif optimizer == "Adam":
            return (torch.optim.Adam, {"lr": lr, "weight_decay": 1e-4})
        else:
            raise TypeError

    def get_before_training_starts_hook(self):
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

    def get_model_kwargs(self, dataset, pretrain_on_src, src_domains):
        # This is None if src_domains == []
        src_domain = main_utils.domain_len_assertion(src_domains)
        doing_uda = not pretrain_on_src
        if dataset.startswith("domainnet"):
            Gkwargs = {"pretrained": True}
            if doing_uda:
                Gkwargs["domain"] = src_domain
            Ckwargs = {"pretrained": doing_uda, "domain": src_domain}
        elif dataset in ["office31", "officehome"]:
            Gkwargs = {"pretrained": True}
            Ckwargs = {"pretrained": doing_uda, "domain": src_domain}
        elif dataset == "mnist":
            Gkwargs = {"pretrained": doing_uda}
            Ckwargs = {"pretrained": doing_uda}
        else:
            raise ValueError

        return Gkwargs, Ckwargs

    def get_models(
        self,
        dataset,
        src_domains,
        pretrain_on_src,
        num_classes,
        feature_layer,
        multilabel,
    ):
        self.num_classes = num_classes
        Gkwargs, Ckwargs = self.get_model_kwargs(dataset, pretrain_on_src, src_domains)
        print("G kwargs", Gkwargs)
        print("C kwargs", Ckwargs)
        G = getattr(pretrained_module, f"{dataset}G")(**Gkwargs)
        C = getattr(pretrained_module, f"{dataset}C")(**Ckwargs)

        models = {"G": G, "C": C}
        models, self.feature_size, framework = self.set_feature_layer(
            models, dataset, pretrain_on_src, feature_layer, multilabel
        )
        models["D"] = Discriminator(in_size=self.feature_size, h=2048)
        return models, framework

    def set_feature_layer(
        self, models, dataset, pretrain_on_src, feature_layer, multilabel
    ):
        fc_out_feature_size = {
            "mnist": 1200,
            "domainnet": 2048,
            "domainnet126": 2048,
            "office31": 2048,
            "officehome": 2048,
        }[dataset]
        framework = IgniteMultiLabelClassification if multilabel else Ignite
        if pretrain_on_src or feature_layer == 0:
            return models, fc_out_feature_size, framework
        if feature_layer in [7, 8]:
            if multilabel:
                raise ValueError(
                    "feature_layer in [7,8] and multilabel=True not currently supported"
                )
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
