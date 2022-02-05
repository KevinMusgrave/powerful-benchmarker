import os

import h5py
import numpy as np
from pytorch_adapt.utils import common_functions as c_f


def get_val_data_hook(folder, is_within_exp_group, exp_name, config_name, trial_name):
    folder = os.path.join(folder, "features")
    c_f.makedir_if_not_there(folder)
    components = os.path.normpath(folder).split(os.path.sep)
    exp_group = components[-4] if is_within_exp_group else ""
    hf = h5py.File(os.path.join(folder, "features.hdf5"), "a")

    def save_features(engine, collected_data):
        epoch = engine.state.epoch
        if epoch == 0:
            return
        all_data = {
            "exp_group": exp_group,
            "exp_name": exp_name,
            "config_name": config_name,
            "trial_name": trial_name,
            "epoch": epoch,
        }
        for k, v in collected_data.items():
            curr_k = k.replace("_with_labels", "")
            all_data.update(
                {
                    f"{curr_k}_{name}": v[name].cpu().numpy()
                    for name in ["features", "logits", "labels"]
                    if name in v
                }
            )
        grp = hf.create_group(f"{epoch}")
        for k, v in all_data.items():
            kwargs = {"compression": "gzip"} if isinstance(v, np.ndarray) else {}
            grp.create_dataset(k, data=v, **kwargs)

    return save_features
