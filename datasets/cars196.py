#! /usr/bin/env python3

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as sio

class Cars196(Dataset):
    def __init__(self, dataset_root, transform=None):
        self.dataset_folder = dataset_root + "/cars196/"
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
        img_data = sio.loadmat("%s/cars_annos.mat" % (self.dataset_folder))
        self.labels = np.array([i[0, 0] for i in img_data["annotations"]["class"][0]])
        self.img_paths = ["%s/%s" % (self.dataset_folder, i[0]) for i in img_data["annotations"]["relative_im_path"][0]]
        self.class_names = [i[0] for i in img_data["class_names"][0]]
