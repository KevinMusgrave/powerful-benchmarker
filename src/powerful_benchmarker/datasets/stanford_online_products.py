#! /usr/bin/env python3

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

class StanfordOnlineProducts(Dataset):
    def __init__(self, dataset_root, transform=None):
        self.dataset_folder = os.path.join(dataset_root, "Stanford_Online_Products")
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
        info_files = {"train": "Ebay_train.txt", "test": "Ebay_test.txt"}
        self.img_paths = []
        self.labels = []
        global_idx = 0
        self.predefined_splits = {}
        for k, v in info_files.items():
            curr_file = read_csv(os.path.join(self.dataset_folder, v), delim_whitespace=True, header=0).values
            self.img_paths.extend([os.path.join(self.dataset_folder, x) for x in list(curr_file[:,3])])
            self.labels.extend(list(curr_file[:,1] - 1))
            self.predefined_splits[k] = np.arange(global_idx, global_idx + len(curr_file))
            global_idx += len(curr_file)
        self.labels = np.array(self.labels)
