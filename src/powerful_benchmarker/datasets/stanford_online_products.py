#! /usr/bin/env python3

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import os
import zipfile
from ..utils import common_functions as c_f

class StanfordOnlineProducts(Dataset):
    url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
    filename = 'Stanford_Online_Products.zip'
    md5 = '7f73d41a2f44250d4779881525aea32e'

    def __init__(self, root, transform=None, download=False):
        self.root = root
        if download:
            self.download_dataset()
        self.dataset_folder = os.path.join(self.root, "Stanford_Online_Products")
        self.load_labels()
        self.transform = transform
        assert len(np.unique(self.labels)) == 22634
        assert self.__len__() == 120053

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

    def download_dataset(self):
        download_url(self.url, self.root, filename=self.filename, md5=self.md5)
        with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zip_ref:
            zip_ref.extractall(self.root, members = c_f.extract_progress(zip_ref))