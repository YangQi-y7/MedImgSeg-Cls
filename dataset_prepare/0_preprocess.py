# note that we didn't transfer much information(e.g. spacing, origin) to the new nii file
# for information contains in a nii file, see in link:

import SimpleITK as sitk
import os
import glob
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from parameters import raw_data_dir, processed_dir

volume_ls = glob.glob(os.path.join('..', raw_data_dir, 'volume-*.nii'))
segmentation_ls = glob.glob(os.path.join('..', raw_data_dir, 'segmentation-*.nii'))


# mkdir
def check_out_dirs(base_dir):
    if not os.path.exists(os.path.join('..', base_dir)):
        os.mkdir(os.path.join('..', base_dir))

    if not os.path.exists(os.path.join('..', base_dir, 'img')):
        os.mkdir(os.path.join('..', base_dir, 'img'))

    if not os.path.exists(os.path.join('..', base_dir, 'mask')):
        os.mkdir(os.path.join('..', base_dir, 'mask'))

check_out_dirs(processed_dir)

# img
for _, volume_dir in enumerate(tqdm(volume_ls)):
    # read img
    img_nii = sitk.ReadImage(volume_dir)
    img = sitk.GetArrayFromImage(img_nii)

    # preprocess
    img_out = np.clip(img, 0, 200)

    # save as nii
    img_nii_out = sitk.GetImageFromArray(img_out)
    name = os.path.split(volume_dir)[-1].split('-')[-1]     # delete prefix
    out_dir = os.path.join('..', processed_dir, 'img', name)
    sitk.WriteImage(img_nii_out, out_dir)

# mask
for _, mask_dir in enumerate(tqdm(segmentation_ls)):
    # read img
    mask_nii = sitk.ReadImage(mask_dir)
    mask = sitk.GetArrayFromImage(mask_nii)

    # extract masks of tumors
    mask_out = (mask == 2).astype(int)

    # save as nii
    mask_nii_out = sitk.GetImageFromArray(mask_out)
    name = os.path.split(mask_dir)[-1].split('-')[-1]     # delete prefix
    out_dir = os.path.join('..', processed_dir, 'mask', name)
    sitk.WriteImage(mask_nii_out, out_dir)

# randomly annotate
annotations = []
for _, volume_dir in enumerate(tqdm(volume_ls)):
    name = os.path.split(volume_dir)[-1].split('-')[-1]  # delete prefix
    label = random.choice([0, 1])
    annotations.append([name, label])

df = pd.DataFrame(data=annotations)
df.columns = ['name', 'label']
df.to_csv(os.path.join('..', processed_dir, 'annotations.csv'))
