from pytorch_adapt.frameworks.ignite import CheckpointFnCreator
from pytorch_adapt.validators import AccuracyValidator, APValidator, ScoreHistory


def get_validator_cls_and_kwargs(num_classes, average, multilabel):
    kwargs = {"num_classes": num_classes, "average": average}
    validator_cls = APValidator if multilabel else AccuracyValidator
    return validator_cls, kwargs


def src_accuracy(num_classes, average, split, multilabel):
    validator_cls, torchmetric_kwargs = get_validator_cls_and_kwargs(
        num_classes, average, multilabel
    )
    return validator_cls(
        torchmetric_kwargs=torchmetric_kwargs, key_map={f"src_{split}": "src_val"}
    )


def target_accuracy(num_classes, average, split, multilabel):
    validator_cls, torchmetric_kwargs = get_validator_cls_and_kwargs(
        num_classes, average, multilabel
    )
    return validator_cls(
        torchmetric_kwargs=torchmetric_kwargs,
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
        validator = target_accuracy(
            num_classes, average="macro", split="train", multilabel=False
        )
    elif validator_name == "oracle_micro":
        validator = target_accuracy(
            num_classes, average="micro", split="train", multilabel=False
        )
    elif validator_name == "src_accuracy":
        validator = src_accuracy(
            num_classes, average="macro", split="val", multilabel=False
        )
    elif validator_name == "oracle_multilabel":
        validator = target_accuracy(
            num_classes, average="macro", split="train", multilabel=True
        )
    elif validator_name == "src_accuracy_multilabel":
        validator = src_accuracy(
            num_classes, average="macro", split="val", multilabel=True
        )
    else:
        raise ValueError

    validator = ScoreHistory(validator, ignore_epoch=0)
    checkpoint_fn = CheckpointFnCreator(dirname=checkpoint_path, require_empty=False)
    return validator, checkpoint_fn
