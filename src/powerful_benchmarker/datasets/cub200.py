#! /usr/bin/env python3

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
import os

class CUB200(Dataset):
    def __init__(self, dataset_root, transform=None):
        img_folder = os.path.join(dataset_root, "cub2011", "CUB_200_2011", "images")
        self.dataset = datasets.ImageFolder(img_folder)
        self.labels = np.array([b for (a, b) in self.dataset.imgs])
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        output_dict = {"data": img, "label": label}
        return output_dict
