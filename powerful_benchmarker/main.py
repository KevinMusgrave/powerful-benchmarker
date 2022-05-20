import json
import os
import shutil
import sys
from functools import partialmethod

from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

import argparse
import logging

logging.basicConfig()
logging.getLogger("pytorch-adapt").setLevel(logging.INFO)

import warnings

warnings.filterwarnings(
    "ignore",
    message="Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.",
)

import joblib
import numpy as np
import optuna
import pytorch_adapt
import torch
from optuna.samplers import PartialFixedSampler, TPESampler
from optuna.trial import TrialState
from pytorch_adapt.frameworks.ignite import Ignite
from pytorch_adapt.frameworks.ignite import utils as ignite_utils
from pytorch_adapt.utils import common_functions as c_f

sys.path.insert(0, ".")
from powerful_benchmarker import configs
from powerful_benchmarker.utils import ignite_save_features, main_utils
from powerful_benchmarker.utils.constants import (
    BEST_TRIAL_FILENAME,
    TRIALS_FILENAME,
    add_default_args,
)
from powerful_benchmarker.utils.get_validator import get_validator
from powerful_benchmarker.utils.logger import Logger

print("pytorch_adapt.__version__", pytorch_adapt.__version__)


def evaluate_best_model(cfg, exp_path):
    assert cfg.validator in ["oracle", "oracle_micro"]
    original_exp_path = exp_path
    with open(os.path.join(exp_path, BEST_TRIAL_FILENAME), "r") as f:
        best_trial = json.load(f)

    exp_path = os.path.join(exp_path, best_trial["number"])
    with open(
        os.path.join(exp_path, "configs", "args_and_trial_params.json"), "r"
    ) as f:
        original_cfg = json.load(f)

    for k in ["dataset", "adapter", "feature_layer"]:
        setattr(cfg, k, original_cfg[k])
    trial = optuna.trial.FixedTrial(original_cfg["trial_params"])

    scores = {}
    for d in cfg.target_domains:
        (
            framework,
            adapter,
            datasets,
            dataloader_creator,
            validator,
            checkpoint_fn,
            _,
            _,
            _,
        ) = get_adapter_datasets_etc(cfg, exp_path, cfg.validator, [d], trial)
        adapter = framework(adapter, checkpoint_fn=checkpoint_fn)
        validator = validator.validator  # don't need ScoreHistory
        scores[d] = main_utils.evaluate(
            adapter, datasets, validator, dataloader_creator
        )

    filename = f"best_model_{cfg.validator}_{'_'.join(cfg.target_domains)}.json"
    with open(os.path.join(original_exp_path, filename), "w") as f:
        json.dump(scores, f, indent=2)


def get_adapter_datasets_etc(
    cfg,
    exp_path,
    validator_name,
    target_domains,
    trial,
    num_fixed_params=0,
):
    if cfg.pretrain_on_src:
        assert cfg.feature_layer == 0
    checkpoint_path = os.path.join(exp_path, "checkpoints")
    num_classes = main_utils.num_classes(cfg.dataset)

    validator, checkpoint_fn = get_validator(
        num_classes,
        validator_name,
        checkpoint_path,
        cfg.feature_layer,
    )

    configerer = getattr(configs, cfg.adapter)(trial)
    datasets = main_utils.get_datasets(
        cfg.dataset,
        cfg.src_domains,
        target_domains,
        cfg.pretrain_on_src,
        cfg.dataset_folder,
        cfg.download_datasets,
    )

    dataloader_creator = main_utils.get_dataloader_creator(
        cfg.batch_size,
        cfg.num_workers,
    )

    models, framework = configerer.get_models(
        dataset=cfg.dataset,
        src_domains=cfg.src_domains,
        start_with_pretrained=cfg.start_with_pretrained,
        pretrain_on_src=cfg.pretrain_on_src,
        num_classes=num_classes,
        feature_layer=cfg.feature_layer,
    )
    optimizers = configerer.get_optimizers(
        cfg.pretrain_on_src, cfg.optimizer, cfg.pretrain_lr
    )
    before_training_starts = configerer.get_before_training_starts_hook(cfg.optimizer)

    adapter = configerer.get_new_adapter(
        models,
        optimizers,
        before_training_starts,
        cfg.lr_multiplier,
        cfg.use_full_inference,
        datasets=datasets,
    )
    logger = Logger(os.path.join(exp_path, "logs"))
    if framework is None:
        framework = Ignite

    if (len(trial.params) - num_fixed_params) > 5:
        raise ValueError("Should only optimize 5 hyperparams")

    return (
        framework,
        adapter,
        datasets,
        dataloader_creator,
        validator,
        checkpoint_fn,
        logger,
        configerer,
        num_classes,
    )


def objective(cfg, root_exp_path, trial, reproduce_iter=None, num_fixed_params=0):
    if reproduce_iter is not None:
        trial_name = f"reproduction{reproduce_iter}"
    else:
        trial_name = str(trial.number)
    exp_path = os.path.join(root_exp_path, trial_name)
    config_path = os.path.join(exp_path, "configs")
    if os.path.isdir(exp_path):
        shutil.rmtree(exp_path)

    (
        framework,
        adapter,
        datasets,
        dataloader_creator,
        validator,
        checkpoint_fn,
        logger,
        configerer,
        num_classes,
    ) = get_adapter_datasets_etc(
        cfg,
        exp_path,
        cfg.validator,
        cfg.target_domains,
        trial,
        num_fixed_params,
    )

    save_features_cls = ignite_save_features.SaveFeatures
    if cfg.adapter == "ATDOCConfig":
        save_features_cls = ignite_save_features.save_features_atdoc(configerer.atdoc)

    val_hooks = main_utils.get_val_hooks(
        cfg, exp_path, logger, num_classes, cfg.pretrain_on_src, save_features_cls
    )

    adapter = framework(
        adapter,
        validator=validator,
        val_hooks=val_hooks,
        checkpoint_fn=checkpoint_fn,
        logger=logger,
        log_freq=1,
    )

    configerer.save(config_path)
    main_utils.save_argparse_and_trial_params(cfg, trial, config_path)
    main_utils.save_this_file(__file__, config_path)

    early_stopper_kwargs = None
    if cfg.patience:
        early_stopper_kwargs = {"patience": cfg.patience}

    best_score, best_epoch = adapter.run(
        datasets=datasets,
        dataloader_creator=dataloader_creator,
        max_epochs=cfg.max_epochs,
        early_stopper_kwargs=early_stopper_kwargs,
        val_interval=cfg.val_interval,
        check_initial_score=cfg.check_initial_score,
    )

    if validator is None:
        if not ignite_utils.is_done(adapter.trainer, cfg.max_epochs):
            return float("nan")
        return 0

    if best_score is None:
        return float("nan")

    return best_score


def hyperparam_search(cfg, exp_path):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_path = os.path.join(exp_path, "study.pkl")
    plot_path = os.path.join(exp_path, "plots")
    log_path = os.path.join(exp_path, TRIALS_FILENAME)

    if os.path.isdir(exp_path) and os.path.isfile(study_path):
        study = joblib.load(study_path)
    else:
        c_f.makedir_if_not_there(exp_path)
        c_f.makedir_if_not_there(plot_path)
        pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            sampler=TPESampler(n_startup_trials=cfg.n_startup_trials),
        )

    num_fixed_params = 0
    if cfg.fixed_param_source:
        fp_source_path = os.path.join(cfg.exp_folder, cfg.fixed_param_source)
        fp_source_best_trial_json = os.path.join(fp_source_path, BEST_TRIAL_FILENAME)
        if not os.path.isfile(fp_source_best_trial_json):
            FileNotFoundError(
                "Fixed param source needs to be complete to use its best params"
            )
        fp_source_path = os.path.join(fp_source_path, "study.pkl")
        fp_source_study = joblib.load(fp_source_path)
        study.sampler = PartialFixedSampler(fp_source_study.best_params, study.sampler)
        num_fixed_params = len(fp_source_study.best_params)

    i = len([st for st in study.trials if st.state == TrialState.COMPLETE])
    print(f"{i} trials already complete")

    study.sampler.reseed_rng()

    while i < cfg.num_trials:
        study.optimize(
            lambda trial: objective(
                cfg, exp_path, trial, num_fixed_params=num_fixed_params
            ),
            n_trials=1,
            timeout=None,
            callbacks=[
                main_utils.save_study(study_path),
                main_utils.plot_visualizations(plot_path),
                main_utils.save_dataframe(log_path),
                main_utils.delete_suboptimal_models(exp_path),
                main_utils.delete_failed_features(exp_path),
            ],
            gc_after_trial=True,
        )
        if study.trials[-1].state == TrialState.COMPLETE:
            print("trial completed successfully, incrementing counter")
            i += 1

    i = main_utils.num_repro_complete(exp_path)
    print("num_reproduce_complete", i)
    while i < cfg.num_reproduce:
        result = objective(
            cfg,
            exp_path,
            optuna.trial.FixedTrial(study.best_trial.params),
            i,
            num_fixed_params=num_fixed_params,
        )
        if not np.isnan(result):
            main_utils.update_repro_file(exp_path)
            i += 1

    best_json = {
        field: str(getattr(study.best_trial, field))
        for field in study.best_trial._ordered_fields
    }
    with open(os.path.join(exp_path, BEST_TRIAL_FILENAME), "w") as f:
        json.dump(best_json, f, indent=2)


def main(cfg):
    exp_path = os.path.join(cfg.exp_folder, cfg.exp_name)
    if cfg.evaluate:
        evaluate_best_model(cfg, exp_path)
    else:
        hyperparam_search(cfg, exp_path)


if __name__ == "__main__":
    print("num gpus available in main =", torch.cuda.device_count())
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder", "dataset_folder"])
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--src_domains", nargs="+", default=[])
    parser.add_argument("--target_domains", nargs="+", default=[])
    parser.add_argument("--adapter", type=str)
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--n_startup_trials", type=int, default=10)
    parser.add_argument("--start_with_pretrained", action="store_true")
    parser.add_argument("--validator", type=str, default=None)
    parser.add_argument("--pretrain_on_src", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--num_reproduce", type=int, default=0)
    parser.add_argument("--feature_layer", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr_multiplier", type=float, default=1)
    parser.add_argument("--pretrain_lr", type=float, default=0.01)
    parser.add_argument("--fixed_param_source", type=str, default=None)
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--download_datasets", action="store_true")
    parser.add_argument("--use_stat_getter", action="store_true")
    parser.add_argument("--check_initial_score", action="store_true")
    parser.add_argument("--use_full_inference", action="store_true")
    args = parser.parse_args()
    main(args)
