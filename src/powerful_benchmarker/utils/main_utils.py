import glob
import json
import os
import shutil
from pathlib import Path

import joblib
import optuna
from pytorch_adapt.datasets import DataloaderCreator
from pytorch_adapt.datasets.getters import (
    get_mnist_mnistm,
    get_office31,
    get_officehome,
)
from pytorch_adapt.frameworks.ignite import IgniteValHookWrapper
from pytorch_adapt.utils import common_functions as c_f
from pytorch_adapt.validators import MultipleValidators, ScoreHistories

from . import get_validator
from .ignite_save_features import SaveFeatures


def save_this_file(file_in, folder):
    if folder is not None:
        c_f.makedir_if_not_there(folder)
        src = Path(file_in).absolute()
        shutil.copyfile(src, os.path.join(folder, os.path.basename(src)))


def save_argparse(cfg, folder):
    if folder is not None:
        c_f.makedir_if_not_there(folder)
        with open(os.path.join(folder, "commandline_args.json"), "w") as f:
            json.dump(cfg.__dict__, f, indent=2)


def reproductions_filename(exp_path):
    return os.path.join(exp_path, "reproduction_score_vs_test_accuracy.csv")


def num_reproductions_complete(exp_path):
    log_path = reproductions_filename(exp_path)
    if not os.path.isfile(log_path):
        return 0
    with open(log_path, "r") as f:
        # subtract 1 for header
        return sum(1 for line in f) - 1


def get_dataloader_creator(batch_size, num_workers):
    return DataloaderCreator(
        train_kwargs={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
        },
        val_kwargs={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
        },
        val_names=[
            "src_train",
            "src_val",
            "target_train",
            "target_val",
            "target_train_with_labels",
            "target_val_with_labels",
        ],
    )


def get_stat_getter(num_classes, pretrain_on_src):
    validators = {
        "src_train_macro": get_validator.src_accuracy(num_classes, split="train"),
        "src_train_micro": get_validator.src_accuracy(
            num_classes, average="micro", split="train"
        ),
        "src_val_macro": get_validator.src_accuracy(num_classes),
        "src_val_micro": get_validator.src_accuracy(num_classes, average="micro"),
    }
    if not pretrain_on_src:
        validators.update(
            {
                "target_train_macro": get_validator.target_accuracy(num_classes),
                "target_train_micro": get_validator.target_accuracy(
                    num_classes, average="micro"
                ),
                "target_val_macro": get_validator.target_accuracy(
                    num_classes, split="val"
                ),
                "target_val_micro": get_validator.target_accuracy(
                    num_classes, average="micro", split="val"
                ),
            }
        )
    for k, v in validators.items():
        assert len(v.required_data) == 1
        assert k.startswith(v.required_data[0].replace("with_labels", ""))
    return ScoreHistories(MultipleValidators(validators=validators))


def get_val_hooks(cfg, folder, logger, num_classes, pretrain_on_src):
    hooks = []
    if cfg.use_stat_getter:
        stat_getter = get_stat_getter(num_classes, pretrain_on_src)
        hooks.append(IgniteValHookWrapper(stat_getter, logger=logger))
    if cfg.save_features:
        hooks.append(SaveFeatures(folder, logger))
    return hooks


def get_datasets(
    dataset,
    src_domains,
    target_domains,
    pretrain_on_src,
    folder,
    download,
):
    if pretrain_on_src and len(target_domains) > 0:
        raise ValueError("target_domain must be [] if pretrain_on_src is True")
    if not set(src_domains).isdisjoint(target_domains):
        raise ValueError(
            f"src_domains {src_domains} and target_domains {target_domains} cannot have any overlap"
        )

    getter = {
        "mnist": get_mnist_mnistm,
        "office31": get_office31,
        "officehome": get_officehome,
    }[dataset]
    datasets = getter(
        src_domains,
        target_domains,
        folder,
        return_target_with_labels=True,
        download=download,
    )
    c_f.LOGGER.info(datasets)
    return datasets


def save_study(study_path):
    def return_func(study, frozen_trial):
        joblib.dump(study, study_path)

    return return_func


def plot_visualizations(plot_path):
    def return_func(study, frozen_trial):
        i = frozen_trial.number
        try:
            fig = optuna.visualization.plot_contour(study)
            fig.write_html(os.path.join(plot_path, f"contour_plot.html"))
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_html(os.path.join(plot_path, f"parallel_coordinate.html"))
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(os.path.join(plot_path, f"importances.html"))
        except:
            pass

    return return_func


def save_dataframe(log_path):
    def return_func(study, frozen_trial):
        study.trials_dataframe().to_csv(log_path, sep=",")

    return return_func


def delete_suboptimal_models(exp_path):
    def return_func(study, frozen_trial):
        print("delete_suboptimal_models")
        try:
            bt = study.best_trial
        except ValueError:
            return
        keep = str(bt.number)
        all_paths = sorted(glob.glob(f"{exp_path}/*"))
        for x in all_paths:
            if os.path.isdir(x):
                trial_name = os.path.basename(x)
                if trial_name.isdigit() and trial_name != keep:
                    model_folder = os.path.join(x, "checkpoints")
                    if os.path.isdir(model_folder):
                        print(f"deleting {model_folder}")
                        shutil.rmtree(model_folder)

    return return_func


# assumes oracle validator
def evaluate(adapter, datasets, validator, dataloader_creator):
    dataloader_creator.all_val = True
    scores = {}
    for split in ["target_train_with_labels", "target_val_with_labels"]:
        validator.key_map = {split: "src_val"}
        scores[split] = adapter.evaluate_best_model(
            datasets, validator, dataloader_creator
        )
    return scores


def num_classes(dataset_name):
    return {
        "mnist": 10,
        "domainnet": 345,
        "domainnet126": 126,
        "office31": 31,
        "officehome": 65,
    }[dataset_name]


def domain_len_assertion(domain_list):
    if len(domain_list) > 1:
        raise ValueError("only 1 domain currently supported")
    return domain_list[0]
