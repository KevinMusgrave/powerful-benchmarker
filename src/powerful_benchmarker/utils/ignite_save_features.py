import os

import h5py
import numpy as np
from pytorch_adapt.utils import common_functions as c_f


def get_val_data_hook(folder, trial_name, logger):
    folder = os.path.join(folder, "features")
    c_f.makedir_if_not_there(folder)

    def save_features(engine, collected_data):
        epoch = engine.state.epoch
        if epoch == 0:
            return

        for k, v in collected_data.items():
            curr_k = k.replace("_with_labels", "")
            inference_dict = {
                curr_k: {
                    name: v[name].cpu().numpy()
                    for name in ["features", "logits", "labels"]
                    if name in v
                }
            }

        losses_dict = logger.get_losses()

        with h5py.File(os.path.join(folder, "features.hdf5"), "a") as hf:
            write_nested_dict(hf, inference_dict, epoch, "inference")
            write_nested_dict(hf, losses_dict, epoch, "losses")

    return save_features


def write_nested_dict(hf, d, epoch, series_name):
    for k1, v1 in d.items():
        grp = hf.create_group(f"{epoch}/{series_name}/{k1}")
        for k2, v2 in v1.items():
            kwargs = (
                {"compression": "gzip"} if isinstance(v2, (np.ndarray, list)) else {}
            )
            grp.create_dataset(k2, data=v2, **kwargs)
