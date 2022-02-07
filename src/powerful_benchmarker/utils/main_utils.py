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


def keys_vals_str(x, correct_header):
    keys = ",".join(str(k) for k in x.keys())
    vals = ",".join(str(k) for k in x.values())
    return keys, vals


def scores_csv_filename(exp_path):
    return os.path.join(exp_path, "score_vs_test_accuracy.csv")


def reproductions_filename(exp_path):
    return os.path.join(exp_path, "reproduction_score_vs_test_accuracy.csv")


def get_scores_csv_filename(exp_path, reproduce_iter=None):
    if reproduce_iter is None:
        return scores_csv_filename(exp_path)
    return reproductions_filename(exp_path)


def num_reproductions_complete(exp_path):
    log_path = reproductions_filename(exp_path)
    if not os.path.isfile(log_path):
        return 0
    with open(log_path, "r") as f:
        # subtract 1 for header
        return sum(1 for line in f) - 1


def write_scores_to_csv(log_path, best_score, accuracies, trial):
    trial_header = "trial,validation_score"
    src_header = "src_train_acc_class_avg,src_train_acc_global,src_val_acc_class_avg,src_val_acc_global"
    target_header = "target_train_acc_class_avg,target_train_acc_global,target_val_acc_class_avg,target_val_acc_global"
    correct_header = f"{trial_header},{src_header},{target_header}"

    keys, vals = keys_vals_str(accuracies, correct_header)
    header = f"{trial_header},{keys}"
    assert header == correct_header
    assert len(header.split(",")) == (2 + len(vals.split(",")))

    write_string = f"{trial},{best_score},{vals}\n"
    if not os.path.isfile(log_path):
        write_string = f"{header}\n{write_string}"
    with open(log_path, "a") as fd:
        fd.write(write_string)


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


def get_accuracies_of_best_model(
    adapter, datasets, saver, dataloader_creator, num_classes
):
    validators = {}
    for domain in ["src", "target"]:
        if domain == "src":
            v_fn = get_validator.src_accuracy
        elif domain == "target":
            v_fn = get_validator.target_accuracy
        else:
            raise ValueError
        for split in ["train", "val"]:
            for average in ["macro", "micro"]:
                # use fresh validator
                validator = v_fn(num_classes, average=average, split=split)
                key = f"{domain}_{split}_acc_"
                key += "class_avg" if average == "macro" else "global"
                validators[key] = validator

    validator = MultipleValidators(validators=validators, return_sub_scores=True)
    _, sub_scores = adapter.evaluate_best_model(
        datasets,
        validator,
        saver,
        dataloader_creator=dataloader_creator,
    )

    return sub_scores


def set_validator_required_data_mapping_for_eval(validator, validator_name, split_name):
    replace_target_train = {split_name: "target_train"}
    if validator_name in ["oracle", "oracle_micro"]:
        validator.key_map = {split_name: "src_val"}
    elif validator_name == "entropy_diversity":
        validator.validators["entropy"].key_map = replace_target_train
        validator.validators["diversity"].key_map = replace_target_train


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
