#! /usr/bin/env python3

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

class CelebA(Dataset):
    def __init__(self, dataset_root, transform=None):
        self.dataset_folder = os.path.join(dataset_root, "celeb_a")
        self.load_labels()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        output_dict = {"data": img, "label": label}
        return output_dict

    def load_labels(self):
        from pandas import read_csv
        splits_info = read_csv(os.path.join(self.dataset_folder, "list_eval_partition.txt"), delim_whitespace=True, header=None).values
        self.img_paths = [os.path.join(self.dataset_folder, "img_align_celeba", x) for x in splits_info[:, 0]]
        attributes = read_csv(os.path.join(self.dataset_folder, "list_attr_celeba.txt"), delim_whitespace=True, header=1)
        self.labels = (attributes.values + 1) // 2
        self.label_names = list(attributes.columns)
        splits = splits_info[:, 1]
        all_idx = np.arange(len(splits))
        self.predefined_splits = {"train": 0, "val": 1, "test": 2}
        self.predefined_splits = {k:all_idx[splits==v] for k,v in self.predefined_splits.items()}