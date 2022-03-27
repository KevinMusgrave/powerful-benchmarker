import copy

import torch
import torch.nn.functional as F


def get_from_hdf5(x, device, key):
    return torch.from_numpy(x[key][()]).to(device)


def get_split_and_layer(x, split, layer, device):
    hdf5_layer = "logits" if layer == "preds" else layer
    features = get_from_hdf5(x, device, f"inference/{split}/{hdf5_layer}")
    if layer == "preds":
        features = F.softmax(features, dim=1)
    return features


def get_full_split_name(domain, split):
    return f"{domain}_{split}"


def get_src_domain(length, device):
    return torch.zeros(length).to(device=device, dtype=torch.long)


def get_target_domain(length, device):
    return torch.ones(length).to(device=device, dtype=torch.long)


def use_src_and_target(x, device, validator, src_split_name, target_split_name, layer):
    src = get_split_and_layer(x, src_split_name, layer, device)
    target = get_split_and_layer(x, target_split_name, layer, device)
    return pass_src_and_target_to_validator(
        validator, src_split_name, target_split_name, layer, src, target, device
    )


def pass_src_and_target_to_validator(
    validator, src_split_name, target_split_name, layer, src, target, device
):
    return validator(
        **{
            src_split_name: {
                layer: src,
                "domain": get_src_domain(len(src), device),
            },
            target_split_name: {
                layer: target,
                "domain": get_target_domain(len(target), device),
            },
        }
    )


class BaseConfig:
    def __init__(self, config):
        self.validator_args = copy.deepcopy(config)
        if self.validator_args.keys() != self.expected_keys():
            raise ValueError(
                f"expected {self.expected_keys()} but got {self.validator_args.keys()}"
            )
        self.split = self.validator_args.get("split", None)
