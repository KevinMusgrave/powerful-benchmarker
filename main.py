import json
import os
import shutil
import sys
from functools import partialmethod

from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

sys.path.insert(0, "src")
import argparse
import logging

logging.basicConfig()
logging.getLogger("pytorch-adapt").setLevel(logging.INFO)

import warnings

warnings.filterwarnings(
    "ignore",
    message="Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.",
)

from collections import defaultdict

import joblib
import numpy as np
import optuna
import pytorch_adapt
import torch
import yaml
from optuna.samplers import PartialFixedSampler, TPESampler
from pytorch_adapt.frameworks.ignite import Ignite, IgniteRecordKeeperLogger
from pytorch_adapt.meta_validators import ForwardOnlyValidator
from pytorch_adapt.utils import common_functions as c_f

from powerful_benchmarker import configs
from powerful_benchmarker.utils import main_utils
from powerful_benchmarker.utils.get_validator import get_validator
from powerful_benchmarker.utils.ignite_save_features import get_val_data_hook

print("pytorch_adapt.__version__", pytorch_adapt.__version__)
assert pytorch_adapt.__version__ == "0.0.41"


def evaluate(cfg, experiment_path, adapter, datasets, validator, saver):
    main_utils.set_validator_required_data_mapping_for_eval(
        validator, cfg.evaluate_validator, cfg.evaluate
    )
    adapter.dist_init()
    dataloader_creator = main_utils.get_dataloader_creator(
        cfg.batch_size, cfg.num_workers
    )
    score = adapter.evaluate_best_model(
        datasets, validator, saver, 0, dataloader_creator=dataloader_creator
    )
    print(validator)
    target_domains = "_".join(k for k in cfg.evaluate_target_domains)
    filename = os.path.join(
        experiment_path,
        f"{cfg.evaluate_validator}_{target_domains}_{cfg.evaluate}_score.txt",
    )
    with open(filename, "w") as fd:
        fd.write(str(score))


def get_adapter_datasets_etc(
    cfg,
    experiment_path,
    validator_name,
    target_domains,
    trial=None,
    config_path=None,
    num_fixed_params=0,
):
    model_save_path = os.path.join(experiment_path, "models")
    stats_save_path = os.path.join(experiment_path, "stats")
    is_evaluation = cfg.evaluate is not None
    num_classes = main_utils.num_classes(cfg.dataset)

    validator, saver = get_validator(
        num_classes,
        validator_name,
        model_save_path,
        stats_save_path,
        cfg.adapter,
        cfg.feature_layer,
    )

    configerer = getattr(configs, cfg.adapter)(trial)
    datasets = main_utils.get_datasets(
        cfg.dataset,
        cfg.src_domains,
        target_domains,
        cfg.pretrain_on_src,
        cfg.dataset_folder,
        is_evaluation,
    )

    models, framework = configerer.get_models(
        dataset=cfg.dataset,
        src_domains=cfg.src_domains,
        start_with_pretrained=cfg.start_with_pretrained,
        pretrain_on_src=cfg.pretrain_on_src,
        num_classes=num_classes,
        feature_layer=cfg.feature_layer,
    )
    if trial is not None:
        optimizers = configerer.get_optimizers(
            cfg.pretrain_on_src, cfg.optimizer_name, cfg.pretrain_lr
        )
        before_training_starts = configerer.get_before_training_starts_hook(
            cfg.optimizer_name
        )
    else:
        optimizers = defaultdict()
        before_training_starts = None

    adapter = configerer.get_new_adapter(
        models,
        optimizers,
        before_training_starts,
        cfg.lr_multiplier,
        datasets=datasets,
    )
    logger_path = os.path.join(experiment_path, "logs")
    logger = IgniteRecordKeeperLogger(folder=logger_path)
    if framework is None:
        framework = Ignite

    if trial is not None and (len(trial.params) - num_fixed_params) > 5:
        raise ValueError("Should only optimize 5 hyperparams")

    return (
        framework,
        adapter,
        datasets,
        validator,
        saver,
        logger,
        configerer,
        num_classes,
    )


def objective(
    cfg, root_experiment_path, trial, reproduce_iter=None, num_fixed_params=0
):
    if reproduce_iter is not None:
        trial_num = f"reproduction{reproduce_iter}"
    else:
        trial_num = str(trial.number)
    experiment_path = os.path.join(root_experiment_path, trial_num)
    config_path = os.path.join(experiment_path, "configs")
    if os.path.isdir(experiment_path):
        shutil.rmtree(experiment_path)

    (
        framework,
        adapter,
        datasets,
        validator,
        saver,
        logger,
        configerer,
        num_classes,
    ) = get_adapter_datasets_etc(
        cfg,
        experiment_path,
        cfg.validator,
        cfg.target_domains,
        trial,
        config_path,
        num_fixed_params,
    )
    dataloader_creator = main_utils.get_dataloader_creator(
        cfg.batch_size,
        cfg.num_workers,
    )
    stat_getter = main_utils.get_stat_getter(num_classes, cfg.pretrain_on_src)

    configerer.save(config_path)
    main_utils.save_argparse(cfg, config_path)
    main_utils.save_this_file(__file__, config_path)

    val_data_hook = None
    if cfg.save_features:
        val_data_hook = get_val_data_hook(os.path.join(experiment_path, "features"))

    adapter = framework(
        adapter,
        validator=validator,
        stat_getter=stat_getter,
        saver=saver,
        logger=logger,
        val_data_hook=val_data_hook,
    )

    meta_validator = ForwardOnlyValidator()

    best_score, best_epoch = meta_validator.run(
        adapter,
        datasets=datasets,
        dataloader_creator=dataloader_creator,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        validation_interval=cfg.validation_interval,
        check_initial_score=True,
    )

    if best_score is None:
        return float("nan")

    scores_csv_filename = main_utils.get_scores_csv_filename(
        root_experiment_path, reproduce_iter
    )
    print("***best score***", best_score)
    accuracies = main_utils.get_accuracies_of_best_model(
        adapter, datasets, saver, dataloader_creator, num_classes
    )
    main_utils.write_scores_to_csv(
        scores_csv_filename,
        best_score,
        accuracies,
        trial_num,
    )
    return best_score


def main(cfg):
    experiment_path = os.path.join(cfg.experiment_path, cfg.experiment_name)
    if cfg.evaluate:
        assert cfg.evaluate in ["target_train_with_labels", "target_val_with_labels"]
        experiment_path = os.path.join(experiment_path, cfg.evaluate_trial)
        (
            framework,
            adapter,
            datasets,
            validator,
            saver,
            _,
            _,
            _,
        ) = get_adapter_datasets_etc(
            cfg, experiment_path, cfg.evaluate_validator, cfg.evaluate_target_domains
        )
        adapter = framework(adapter)
        evaluate(cfg, experiment_path, adapter, datasets, validator, saver)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_path = os.path.join(experiment_path, "study.pkl")
        plot_path = os.path.join(experiment_path, "plots")
        log_path = os.path.join(experiment_path, "trials.csv")

        if os.path.isdir(experiment_path) and os.path.isfile(study_path):
            study = joblib.load(study_path)
        else:
            c_f.makedir_if_not_there(experiment_path)
            c_f.makedir_if_not_there(plot_path)
            pruner = optuna.pruners.NopPruner()
            study = optuna.create_study(
                direction="maximize",
                pruner=pruner,
                sampler=TPESampler(n_startup_trials=cfg.n_startup_trials),
            )

        num_fixed_params = 0
        if cfg.fixed_param_source:
            fp_source_path = cfg.fixed_param_source
            fp_source_best_trial_json = os.path.join(fp_source_path, "best_trial.json")
            if not os.path.isfile(fp_source_best_trial_json):
                FileNotFoundError(
                    "Fixed param source needs to be complete to use its best params"
                )
            fp_source_path = os.path.join(fp_source_path, "study.pkl")
            fp_source_study = joblib.load(fp_source_path)
            study.sampler = PartialFixedSampler(
                fp_source_study.best_params, study.sampler
            )
            num_fixed_params = len(fp_source_study.best_params)

        i = len([st for st in study.trials if st.value is not None])

        study.sampler.reseed_rng()

        while i < cfg.num_trials:
            study.optimize(
                lambda trial: objective(
                    cfg, experiment_path, trial, num_fixed_params=num_fixed_params
                ),
                n_trials=1,
                timeout=None,
                callbacks=[
                    main_utils.save_study(study_path),
                    main_utils.plot_visualizations(plot_path),
                    main_utils.save_dataframe(log_path),
                    main_utils.delete_suboptimal_models(experiment_path),
                ],
                gc_after_trial=True,
            )
            if study.trials[-1].value is not None:
                i += 1

        i = main_utils.num_reproductions_complete(experiment_path)
        print("num_reproduce_complete", i)
        while i < cfg.num_reproduce:
            result = objective(
                cfg,
                experiment_path,
                optuna.trial.FixedTrial(study.best_trial.params),
                i,
                num_fixed_params=num_fixed_params,
            )
            if not np.isnan(result):
                i += 1

        best_json = {
            field: str(getattr(study.best_trial, field))
            for field in study.best_trial._ordered_fields
        }
        with open(os.path.join(experiment_path, "best_trial.json"), "w") as f:
            json.dump(best_json, f, indent=2)


if __name__ == "__main__":
    print("num gpus available in main =", torch.cuda.device_count())

    with open("constants.yaml", "r") as f:
        constants = yaml.safe_load(f)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--src_domains", nargs="+", required=True)
    parser.add_argument("--target_domains", nargs="+", required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument(
        "--experiment_path", type=str, default=constants["experiment_folder"]
    )
    parser.add_argument(
        "--dataset_folder", type=str, default=constants["dataset_folder"]
    )
    parser.add_argument("--experiment_name", type=str, default="test")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--n_startup_trials", type=int, default=10)
    parser.add_argument("--start_with_pretrained", action="store_true")
    parser.add_argument("--validator", type=str, default="oracle")
    parser.add_argument("--pretrain_on_src", action="store_true")
    parser.add_argument("--evaluate", type=str, default=None)
    parser.add_argument("--evaluate_trial", type=str, default=None)
    parser.add_argument("--evaluate_validator", type=str, default=None)
    parser.add_argument("--evaluate_target_domains", nargs="+", default=None)
    parser.add_argument("--num_reproduce", type=int, default=0)
    parser.add_argument("--feature_layer", type=int, default=0)
    parser.add_argument("--optimizer_name", type=str, default="SGD")
    parser.add_argument("--lr_multiplier", type=float, default=1)
    parser.add_argument("--pretrain_lr", type=float, default=0.01)
    parser.add_argument("--fixed_param_source", type=str, default=None)
    parser.add_argument("--save_features", action="store_true")
    args = parser.parse_args()
    main(args)
