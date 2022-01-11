import os

import torch
from pytorch_adapt.utils import common_functions as c_f


def get_val_data_hook(folder):
    c_f.makedir_if_not_there(folder)

    def save_features(engine, collected_data):
        epoch = engine.state.epoch
        for k, v in collected_data.items():
            if k not in ["src_val", "target_val_with_labels"]:
                continue
            for feature_name in ["logits", "labels"]:
                filename = os.path.join(folder, f"{k}_{feature_name}_{epoch}.pt")
                torch.save(v[feature_name].cpu(), filename)

    return save_features
