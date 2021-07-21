import os
from parameters import patches_dir, png_dir
import SimpleITK as sitk
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as dataset
import cv2

base_dir = png_dir



def get_img(img_dir, shape):
    if shape == '3D':
        img_nii = sitk.ReadImage(img_dir)
        img = sitk.GetArrayFromImage(img_nii)
    elif shape == '2D':
        img = cv2.imread(img_dir, 0)
        img //= 255
    return img


def normalization(img):
    max_ = np.max(img)
    min_ = np.min(img)
    return (img - min_) / (max_ - min_)


class Dataset(dataset):
    def __init__(self, shape, test=False):
        self.test = test
        self.shape = shape
        if shape == '3D':
            base_dir = patches_dir
        elif shape == '2D':
            base_dir = png_dir
        else:
            print('No such shape.')
            exit(0)

        annotations = pd.read_csv(os.path.join(base_dir, 'annotations.csv'), index_col=0, header=0).values
        self.annotations = dict(zip(annotations[:, 0], annotations[:, 1]))
        if self.test:
            self.img_list = glob.glob(os.path.join(base_dir, 'test', 'img', '*'))
            self.mask_list = glob.glob(os.path.join(base_dir, 'test', 'mask', '*'))
        else:
            self.img_list = glob.glob(os.path.join(base_dir, 'train', 'img', '*'))
            self.mask_list = glob.glob(os.path.join(base_dir, 'train', 'mask', '*'))

    def __getitem__(self, index):
        # get img
        img = get_img(self.img_list[index], self.shape)
        mask = get_img(self.mask_list[index], self.shape)
        # get label
        name = os.path.split(self.img_list[index])[-1]
        label = self.annotations[name]

        # preprocessing
        img = normalization(img)

        # to tensor
        img = torch.FloatTensor(img).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)
        label = label.to(torch.float32)

        # return
        if self.test:
            return img, mask, label, name
        else:
            return img, mask, label

    def __len__(self):
        return len(self.img_list)
