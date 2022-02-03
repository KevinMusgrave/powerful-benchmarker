import os

import numpy as np
from pytorch_adapt.utils import common_functions as c_f


def get_val_data_hook(folder):
    c_f.makedir_if_not_there(folder)

    def save_features(engine, collected_data):
        epoch = engine.state.epoch
        if epoch == 0:
            return
        all_data = {"epoch": epoch}
        for k, v in collected_data.items():
            curr_k = k.replace("_with_labels", "")
            all_data.update(
                {
                    f"{curr_k}_{name}": v[name].cpu().numpy()
                    for name in ["features", "logits", "labels"]
                    if name in v
                }
            )
        filename = os.path.join(folder, f"features_{epoch}")
        np.savez_compressed(filename, **all_data)

    return save_features
