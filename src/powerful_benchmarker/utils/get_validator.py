import os

import torch
from pytorch_adapt.frameworks.ignite import savers
from pytorch_adapt.layers import NLLLoss
from pytorch_adapt.validators import (
    AccuracyValidator,
    DeepEmbeddedValidator,
    DiversityValidator,
    EntropyValidator,
    MultipleValidators,
    ScoreHistory,
    SNDValidator,
)


def src_accuracy(num_classes, average="macro", split="val"):
    kwargs = {"num_classes": num_classes, "average": average}
    return AccuracyValidator(
        torchmetric_kwargs=kwargs, key_map={f"src_{split}": "src_val"}
    )


def target_accuracy(num_classes, average="macro", split="train"):
    kwargs = {"num_classes": num_classes, "average": average}
    return AccuracyValidator(
        torchmetric_kwargs=kwargs,
        key_map={f"target_{split}_with_labels": "src_val"},
    )


def get_validator(
    num_classes,
    validator_name,
    model_save_path,
    stats_save_path,
    adapter_config_name,
    feature_layer,
):
    if validator_name is None:
        return None, None
    adapter_saver = savers.AdapterSaver(folder=model_save_path)
    saver = savers.Saver(adapter_saver=adapter_saver, folder=stats_save_path)

    if validator_name == "oracle":
        validator = target_accuracy(num_classes)
    elif validator_name == "oracle_micro":
        validator = target_accuracy(num_classes, average="micro")
    elif validator_name == "src_accuracy":
        validator = src_accuracy(num_classes)
    elif validator_name == "entropy_diversity":
        layer = "preds" if feature_layer == 8 else "logits"
        validator = MultipleValidators(
            validators={
                "entropy": EntropyValidator(layer=layer),
                "diversity": DiversityValidator(layer=layer),
            },
        )
    elif validator_name == "SND":
        validator = SNDValidator()
    elif validator_name in ["DEV", "DEV_binary"]:
        if validator_name == "DEV":
            if feature_layer == 8:
                error_fn = NLLLoss(reduction="none")
                error_layer = "preds"
            else:
                error_fn = torch.nn.CrossEntropyLoss(reduction="none")
                error_layer = "logits"
        elif validator_name == "DEV_binary":
            error_fn = dev_binary_fn
            error_layer = "logits"
        temp_folder = os.path.join(model_save_path, "val_temp_folder")
        validator = DeepEmbeddedValidator(
            temp_folder=temp_folder,
            num_workers=0,
            batch_size=256,
            error_fn=error_fn,
            error_layer=error_layer,
        )
    # elif validator_name == "KNN":
    #     validator = KNNValidator()

    validator = ScoreHistory(validator, ignore_epoch=0)
    return validator, saver


def dev_binary_fn(preds, labels):
    preds = torch.argmax(preds, dim=1)
    return (preds != labels).float()
