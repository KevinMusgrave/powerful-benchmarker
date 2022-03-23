import torch
from torchmetrics.functional import accuracy

from .base_config import (
    BaseConfig,
    get_from_hdf5,
    get_full_split_name,
    get_src_domain,
    get_target_domain,
)


class DLogitsAccuracy(BaseConfig):
    def score(self, x, exp_config, device):
        src_split = get_full_split_name("src", self.split)
        target_split = get_full_split_name("target", self.split)

        src_logits = get_from_hdf5(x, device, f"inference/{src_split}/d_logits")
        target_logits = get_from_hdf5(x, device, f"inference/{target_split}/d_logits")

        src_preds = torch.sigmoid(src_logits)
        target_preds = torch.sigmoid(target_logits)
        src_labels = get_src_domain(len(src_preds), device)
        target_labels = get_target_domain(len(target_preds), device)

        preds = torch.cat([src_preds, target_preds], dim=0)
        labels = torch.cat([src_labels, target_labels], dim=0).to(
            dtype=torch.long, device=preds.device
        )

        return accuracy(
            preds,
            labels,
            average="macro",
            num_classes=2,
            multiclass=True,
        ).item()
