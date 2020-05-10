#! /usr/bin/env python3

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as sio
from torchvision.datasets.utils import download_url
import os
import tarfile
from ..utils import common_functions as c_f

class Cars196(Dataset):
    ims_url = 'http://imagenet.stanford.edu/internal/car196/car_ims.tgz'
    ims_filename = 'car_ims.tgz'
    ims_md5 = 'd5c8f0aa497503f355e17dc7886c3f14'

    annos_url = 'http://imagenet.stanford.edu/internal/car196/cars_annos.mat'
    annos_filename = 'cars_annos.mat'
    annos_md5 = 'b407c6086d669747186bd1d764ff9dbc'

    def __init__(self, root, transform=None, download=False):
        self.root = os.path.join(root, "cars196")
        if download:
            self.download_dataset()
        self.dataset_folder = self.root
        self.load_labels()
        self.transform = transform
        assert len(np.unique(self.labels)) == 196
        assert self.__len__() == 16185

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
        img_data = sio.loadmat(os.path.join(self.dataset_folder, "cars_annos.mat"))
        self.labels = np.array([i[0, 0] for i in img_data["annotations"]["class"][0]])
        self.img_paths = [os.path.join(self.dataset_folder, i[0]) for i in img_data["annotations"]["relative_im_path"][0]]
        self.class_names = [i[0] for i in img_data["class_names"][0]]

    def download_dataset(self):
        url_infos = [(self.ims_url, self.ims_filename, self.ims_md5), 
                    (self.annos_url, self.annos_filename, self.annos_md5)]
        for url, filename, md5 in url_infos:
            download_url(url, self.root, filename=filename, md5=md5)
        with tarfile.open(os.path.join(self.root, self.ims_filename), "r:gz") as tar:
            tar.extractall(path=self.root, members = c_f.extract_progress(tar))