import os
import numpy as np
import SimpleITK as sitk
import random
import pandas as pd
from tqdm import tqdm
from parameters import K, processed_dir, patches_dir, patch_size


def check_out_dirs(base_dir):
    if not os.path.exists(os.path.join('..', base_dir)):
        os.mkdir(os.path.join('..', base_dir))

    if not os.path.exists(os.path.join('..', base_dir, 'img')):
        os.mkdir(os.path.join('..', base_dir, 'img'))

    if not os.path.exists(os.path.join('..', base_dir, 'mask')):
        os.mkdir(os.path.join('..', base_dir, 'mask'))


def get_img(base_dir, name):
    img_dir = os.path.join('..', base_dir, name)
    img_nii = sitk.ReadImage(img_dir)
    img = sitk.GetArrayFromImage(img_nii)
    return img


def check_boundary(shape, center, a, b, c):
    center[0] = np.clip(center[0], 0 + a, shape[0] - a)
    center[1] = np.clip(center[1], 0 + b, shape[1] - b)
    center[2] = np.clip(center[2], 0 + c, shape[1] - c)
    return center


def save_patch(img, center, out_dir):
    a, b, c = patch_size[0]//2, patch_size[1]//2, patch_size[2]//2
    center = check_boundary(img.shape, center, a, b, c)
    img_patch = img[center[0]-a:center[0]+a, center[1]-b:center[1]+b, center[2]-c:center[2]+c]

    img_out = sitk.GetImageFromArray(img_patch)
    sitk.WriteImage(img_out, out_dir)


def get_patches(img_dir, mask_dir):
    for root, dirs, files in os.walk('../' + mask_dir):
        for _, name in enumerate(tqdm(files)):

            # get_images
            img = get_img(img_dir, name)
            mask = get_img(mask_dir, name)

            # get tumor points list
            tumor_ls = []
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    for k in range(mask.shape[2]):
                        if mask[i][j][k] == 1:
                            tumor_ls.append([i, j, k])

            # randomly choose patch center
            centers = random.choices(tumor_ls, k=K)
            count = 1
            for center in centers:
                new_name = name.split('.')[0] + '_' + str(count) + '.nii'

                # save patches
                out_dir = os.path.join('..', patches_dir, 'img', new_name)
                save_patch(img, center, out_dir)

                out_dir = os.path.join('..', patches_dir, 'Mask', new_name)
                save_patch(mask, center, out_dir)

                count += 1


def update_annotation():
    annotations = pd.read_csv(os.path.join('..', processed_dir, 'annotations.csv'), header=0, index_col=0)
    new_annotations = []

    for [name, label] in annotations.values:
        for i in range(1, K+1):
            new_name = name.split('.')[0] + '_' + str(i) + '.nii'
            new_annotations.append([new_name, label])

    df = pd.DataFrame(data=new_annotations)
    df.columns = ['name', 'label']
    df.to_csv(os.path.join('..', patches_dir, 'annotations.csv'))


if __name__ == '__main__':
    img_dir = os.path.join(processed_dir, 'img')
    mask_dir = os.path.join(processed_dir, 'mask')

    check_out_dirs(patches_dir)
    get_patches(img_dir, mask_dir)

    update_annotation()


