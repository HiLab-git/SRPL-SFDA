# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import random
from glob import glob

import h5py
import torch
import itertools
import numpy as np
from scipy import ndimage
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from scipy.ndimage.interpolation import zoom

class BaseDataSet(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None , Domain_args = 'target'):
        self._base_dir = base_dir
        self.image_list = []
        self.split = split
        self.transform = transform
        self.Domain_args = Domain_args
        if split=='train':
            with open(self._base_dir + '/data_split/' + Domain_args + '/image_train_slice.csv', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            with open(self._base_dir + '/data_split/' + Domain_args + '/image_valid.csv', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir + '/data_split/' + Domain_args + '/image_test.csv', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        # split_data 的作用
        if num is not None and self.split == "train":
            self.image_list = self.image_list[:num]
        print("There is total {0:} samples for {1:}".format(len(self.image_list) , self.split))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx ):
        image_name = self.image_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/fetal_brain_norm_SAM_PL/fetal_brain_norm_slices/setting1_NL_trans/" + self.Domain_args +
                            "/{}.h5".format(image_name), 'r')
        elif self.split == 'val':
            h5f = h5py.File(self._base_dir + "/fetal_brain_norm_SAM_PL/fetal_brain_norm_volumes/setting1_NL_trans/" + self.Domain_args +
                            "/{}.h5".format(image_name), 'r')
        elif self.split == 'test':
            h5f = h5py.File(self._base_dir + "/fetal_brain_norm_SAM_PL/fetal_brain_norm_volumes/setting1_NL_trans/" + self.Domain_args +
                            "/{}.h5".format(image_name), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        gt = h5f['gt'][:]
        uncertainty_map = h5f['uncertainty_map'][:]
        sample = {'image': image, 'label': label , 'gt': gt , 'uncertainty_map': uncertainty_map}
        sample["idx"] = idx
        sample["image_name"] = image_name
        sample = {"image": image, "label": label, "gt": gt, "uncertainty_map": uncertainty_map, "image_name": image_name, "idx": idx}

        if self.split == "train":
            if self.transform:
                sample = self.transform(sample)
        if self.split == "val":
            if self.transform:
                sample = self.transform(sample)
        return sample
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}
class TrainToTensor(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        gt, uncertainty_map = sample["gt"], sample["uncertainty_map"]
        image_name = sample["image_name"]
        idx = sample["idx"]
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        gt = zoom(
            gt, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        uncertainty_map = zoom(
            uncertainty_map, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        gt = torch.from_numpy(gt.astype(np.uint8))
        uncertainty_map = torch.from_numpy(uncertainty_map.astype(np.float32))
        sample = {"image": image, "label": label, "gt": gt,
                  "uncertainty_map": uncertainty_map, "image_name": image_name, "idx": idx}
        return sample


class ValToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt = sample["image"], sample["gt"]
        label, uncertainty_map = sample["label"], sample["uncertainty_map"]
        image_name = sample["image_name"]
        idx = sample["idx"]
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        gt = torch.from_numpy(gt.astype(np.uint8))
        uncertainty_map = torch.from_numpy(uncertainty_map.astype(np.float32))
        sample = {"image": image, "label": label, "gt": gt, "uncertainty_map": uncertainty_map, "image_name": image_name, "idx": idx}
        return sample
    