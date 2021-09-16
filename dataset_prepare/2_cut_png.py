import os
import numpy as np
import SimpleITK as sitk
import random
import pandas as pd
from tqdm import tqdm
import glob
import cv2
from parameters import K, patches_dir, png_dir, patch_size


def check_out_dirs(base_dir):
    if not os.path.exists(os.path.join('..', base_dir)):
        os.mkdir(os.path.join('..', base_dir))

    if not os.path.exists(os.path.join('..', base_dir, 'img')):
        os.mkdir(os.path.join('..', base_dir, 'img'))

    if not os.path.exists(os.path.join('..', base_dir, 'mask')):
        os.mkdir(os.path.join('..', base_dir, 'mask'))


def get_data(img_dir):
    img_nii = sitk.ReadImage(img_dir)
    img = sitk.GetArrayFromImage(img_nii)

    mask_dir = img_dir.replace('img', 'mask')
    mask_nii = sitk.ReadImage(mask_dir)
    mask = sitk.GetArrayFromImage(mask_nii)

    name = os.path.split(img_dir)[-1]
    label = annotations[name]

    return img, mask, label, name


def normalization(img):
    max_ = np.max(img)
    min_ = np.min(img)
    return (img - min_) / (max_ - min_)


def extract_and_save_slices(img, mask, name, label):
    additional_annotations = []
    i = 1
    for slice, msk in zip(img, mask):
        if msk.any():
            new_name = name.split('.nii')[0] + '_'+ str(i) + '.png'
            additional_annotations.append([new_name, label])

            # normalization
            slice = normalization(slice) * 255
            msk = msk * 255
            # save
            slice_dir = os.path.join('..', png_dir, 'img', new_name)
            msk_dir = os.path.join('..', png_dir, 'mask', new_name)
            cv2.imwrite(slice_dir, slice)
            cv2.imwrite(msk_dir, msk)
            i += 1

    return additional_annotations


def update_annotations(new_annotations):
    df = pd.DataFrame(data=new_annotations)
    df.columns = ['name', 'label']
    df.to_csv(os.path.join('..', png_dir, 'annotations.csv'))


def cut_png():
    new_annotations = []
    for img_dir in img_ls:
        img, mask, label, name = get_data(img_dir)
        additional_annotations = extract_and_save_slices(img, mask, name, label)
        new_annotations.extend(additional_annotations)

    update_annotations(new_annotations)


if __name__ == '__main__':
    check_out_dirs(png_dir)

    img_ls = glob.glob(os.path.join('..', patches_dir, 'train', 'img', '*.nii'))
    annotations = pd.read_csv(os.path.join('..', patches_dir, 'annotations.csv'), header=0, index_col=0).values
    annotations = dict(zip(annotations[:, 0], annotations[:, 1]))
    cut_png()
    
    img_ls = glob.glob(os.path.join('..', patches_dir, 'test', 'img', '*.nii'))
    cut_png()
