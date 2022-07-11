import glob
import json
import os
import shutil
from pathlib import Path

import joblib
import optuna
from optuna.trial import TrialState
from pytorch_adapt.datasets import DataloaderCreator
from pytorch_adapt.datasets.getters import (
    get_domainnet126,
    get_mnist_mnistm,
    get_office31,
    get_officehome,
)
from pytorch_adapt.frameworks.ignite import IgniteValHookWrapper
from pytorch_adapt.utils import common_functions as c_f
from pytorch_adapt.validators import MultipleValidators, ScoreHistories

from . import get_validator


def save_this_file(file_in, folder):
    if folder is not None:
        c_f.makedir_if_not_there(folder)
        src = Path(file_in).absolute()
        shutil.copyfile(src, os.path.join(folder, os.path.basename(src)))


def save_argparse_and_trial_params(cfg, trial, folder):
    if folder is not None:
        c_f.makedir_if_not_there(folder)
        with open(os.path.join(folder, "args_and_trial_params.json"), "w") as f:
            dict_to_save = {
                **cfg.__dict__,
                "trial_params": trial.params,
                "trial_num": trial.number,
            }
            json.dump(dict_to_save, f, indent=2)


def update_repro_file(exp_path):
    x, filepath = num_repro_complete(exp_path, return_filepath=True)
    x += 1
    with open(filepath, "w") as f:
        json.dump({"num_repro": x}, f, indent=2)


def num_repro_complete(exp_path, return_filepath=False):
    filepath = os.path.join(exp_path, "num_repro_complete.json")
    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            x = json.load(f)["num_repro"]
    else:
        x = 0
    if return_filepath:
        return x, filepath
    return x


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


def get_val_hooks(cfg, folder, logger, num_classes, pretrain_on_src, save_features_cls):
    hooks = []
    if cfg.use_stat_getter:
        stat_getter = get_stat_getter(num_classes, pretrain_on_src)
        hooks.append(IgniteValHookWrapper(stat_getter, logger=logger))
    if cfg.save_features:
        hooks.append(save_features_cls(folder, logger))
    return hooks


def get_datasets(
    dataset,
    src_domains,
    target_domains,
    pretrain_on_src,
    folder,
    download,
    evaluate,
):
    if not evaluate and pretrain_on_src and len(target_domains) > 0:
        raise ValueError("target_domain must be [] if pretrain_on_src is True")
    if not set(src_domains).isdisjoint(target_domains):
        raise ValueError(
            f"src_domains {src_domains} and target_domains {target_domains} cannot have any overlap"
        )

    getter = {
        "mnist": get_mnist_mnistm,
        "office31": get_office31,
        "officehome": get_officehome,
        "domainnet126": get_domainnet126,
    }[dataset]
    datasets = getter(
        src_domains,
        target_domains,
        folder,
        return_target_with_labels=True,
        download=download,
    )
    c_f.LOGGER.debug(datasets)
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
            print("no best_trial yet")
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


def delete_failed_features(exp_path):
    def return_func(study, frozen_trial):
        print("delete_failed_features")
        for st in study.trials:
            if st.state == TrialState.COMPLETE:
                continue
            features_folder = os.path.join(exp_path, str(st.number), "features")
            if os.path.isdir(features_folder):
                print(f"deleting {features_folder}")
                shutil.rmtree(features_folder)

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
    if len(domain_list) == 0:
        return None
    return domain_list[0]
