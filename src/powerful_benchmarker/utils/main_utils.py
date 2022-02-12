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
from pytorch_adapt.utils import common_functions as c_f
from pytorch_adapt.validators import MultipleValidators, ScoreHistories

from . import get_validator


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
        "src_train_acc_class_avg": get_validator.src_accuracy(
            num_classes, split="train"
        ),
        "src_train_acc_global": get_validator.src_accuracy(
            num_classes, average="micro", split="train"
        ),
        "src_val_acc_class_avg": get_validator.src_accuracy(num_classes),
        "src_val_acc_global": get_validator.src_accuracy(num_classes, average="micro"),
    }
    if not pretrain_on_src:
        validators.update(
            {
                "target_train_acc_class_avg": get_validator.target_accuracy(
                    num_classes
                ),
                "target_train_acc_global": get_validator.target_accuracy(
                    num_classes, average="micro"
                ),
                "target_val_acc_class_avg": get_validator.target_accuracy(
                    num_classes, split="val"
                ),
                "target_val_acc_global": get_validator.target_accuracy(
                    num_classes, average="micro", split="val"
                ),
            }
        )
    for k, v in validators.items():
        assert len(v.required_data) == 1
        assert k.startswith(v.required_data[0].replace("with_labels", ""))
    return ScoreHistories(MultipleValidators(validators=validators))


def get_datasets(
    dataset,
    src_domains,
    target_domains,
    pretrain_on_src,
    folder,
    download,
    is_evaluation=False,
):
    if (not is_evaluation) and (not set(src_domains).isdisjoint(target_domains)):
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
    if pretrain_on_src:
        datasets["train"] = datasets["train"].source_dataset
        if not is_evaluation:
            datasets = {
                k: v
                for k, v in datasets.items()
                if k in ["train", "src_train", "src_val"]
            }
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
                    model_folder = os.path.join(x, "models")
                    if os.path.isdir(model_folder):
                        print(f"deleting {model_folder}")
                        shutil.rmtree(model_folder)

    return return_func


def set_validator_required_data_mapping_for_eval(validator, validator_name, split_name):
    replace_target_train = {split_name: "target_train"}
    if validator_name in ["oracle", "oracle_micro"]:
        validator.key_map = {split_name: "src_val"}
    elif validator_name == "entropy_diversity":
        validator.validators["entropy"].key_map = replace_target_train
        validator.validators["diversity"].key_map = replace_target_train


def evaluate(cfg, exp_path, adapter, datasets, validator, saver):
    set_validator_required_data_mapping_for_eval(
        validator, cfg.evaluate_validator, cfg.evaluate
    )
    adapter.dist_init()
    dataloader_creator = get_dataloader_creator(cfg.batch_size, cfg.num_workers)
    score = adapter.evaluate_best_model(
        datasets, validator, saver, 0, dataloader_creator=dataloader_creator
    )
    print(validator)
    target_domains = "_".join(k for k in cfg.evaluate_target_domains)
    filename = os.path.join(
        exp_path,
        f"{cfg.evaluate_validator}_{target_domains}_{cfg.evaluate}_score.txt",
    )
    with open(filename, "w") as fd:
        fd.write(str(score))


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
