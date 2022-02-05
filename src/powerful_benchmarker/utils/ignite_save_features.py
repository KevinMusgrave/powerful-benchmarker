import os

import h5py
import numpy as np
from pytorch_adapt.utils import common_functions as c_f


def get_val_data_hook(cfg, folder, trial_name):
    folder = os.path.join(folder, "features")
    c_f.makedir_if_not_there(folder)
    components = os.path.normpath(folder).split(os.path.sep)

    def save_features(engine, collected_data):
        epoch = engine.state.epoch
        if epoch == 0:
            return

        all_data = {
            k: getattr(cfg, k)
            for k in [
                "dataset",
                "src_domains",
                "target_domains",
                "adapter",
                "exp_name",
                "optimizer",
                "max_epochs",
                "patience",
                "validation_interval",
                "batch_size",
                "start_with_pretrained",
                "feature_layer",
                "lr_multiplier",
            ]
        }

        all_data.update(
            {
                "trial_name": trial_name,
                "epoch": epoch,
            }
        )

        for k, v in collected_data.items():
            curr_k = k.replace("_with_labels", "")
            all_data.update(
                {
                    f"{curr_k}_{name}": v[name].cpu().numpy()
                    for name in ["features", "logits", "labels"]
                    if name in v
                }
            )

        with h5py.File(os.path.join(folder, "features.hdf5"), "a") as hf:
            grp = hf.create_group(f"{epoch}")
            for k, v in all_data.items():
                kwargs = (
                    {"compression": "gzip"} if isinstance(v, (np.ndarray, list)) else {}
                )
                grp.create_dataset(k, data=v, **kwargs)

    return save_features
