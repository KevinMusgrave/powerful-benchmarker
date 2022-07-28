from pytorch_adapt.frameworks.ignite import CheckpointFnCreator
from pytorch_adapt.validators import AccuracyValidator, ScoreHistory


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
    checkpoint_path,
):
    if validator_name is None:
        return None, None

    if validator_name == "oracle":
        validator = target_accuracy(num_classes)
    elif validator_name == "oracle_micro":
        validator = target_accuracy(num_classes, average="micro")
    elif validator_name == "src_accuracy":
        validator = src_accuracy(num_classes)

    validator = ScoreHistory(validator, ignore_epoch=0)
    checkpoint_fn = CheckpointFnCreator(dirname=checkpoint_path, require_empty=False)
    return validator, checkpoint_fn
