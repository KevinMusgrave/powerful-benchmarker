#! /usr/bin/env python3

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class StanfordOnlineProducts(Dataset):
    def __init__(self, dataset_root, transform=None):
        self.dataset_folder = dataset_root + "/Stanford_Online_Products/"
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
        info_files = {"train": "Ebay_train.txt", "test": "Ebay_test.txt"}
        self.img_paths = []
        self.labels = []
        global_idx = 0
        self.predefined_split_idx = {}
        for k, v in info_files.items():
            local_idx = 0
            with open(self.dataset_folder + "/" + v, "r") as f:
                for iii, line in enumerate(f):
                    if iii > 0:
                        local_idx += 1
                        split_line = line.split(" ")
                        self.img_paths.append(
                            self.dataset_folder + "/" + split_line[3].rstrip("\n")
                        )
                        self.labels.append(int(split_line[1].rstrip("\n")) - 1)
            self.predefined_split_idx[k] = np.arange(global_idx, global_idx + local_idx)
            global_idx += local_idx
        self.labels = np.array(self.labels)
