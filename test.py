import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import SimpleITK as sitk
from metrics.metrics import metrics
import pandas as pd
import datetime
import cv2
from parameters import batch_size


def get_model(id):
    if id == 'seg_3D':
        from net.Unet_3D import Unet
        model = Unet(1, 1).cuda()
        model_dir = 'saved_models/seg_3D/net95-0.608.pth'

        from dataset_prepare.dataset import Dataset
        test_ds = Dataset(shape='3D', test=True)
        test_dl = DataLoader(test_ds, batch_size=1)

    elif id == 'seg':
        from net.Unet_2D import Unet
        model = Unet(1, 1).cuda()
        model_dir = 'saved_models/seg/net100-0.378.pth'

        from dataset_prepare.dataset import Dataset
        test_ds = Dataset(shape='2D', test=True)
        test_dl = DataLoader(test_ds, batch_size=1)

    elif id == 'cls':
        from net.pre_trained_models import ResNet18
        model = ResNet18(1).cuda()
        model_dir = 'saved_models/cls/net95-43.182.pth'

        from dataset_prepare.dataset import Dataset
        test_ds = Dataset(shape='2D', test=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size)

    elif id == 'multi':
        from net.multi_task_model import MultiTaskModel
        model = MultiTaskModel().cuda()
        model_dir = 'saved_models/multi/net.pth'

        from dataset_prepare.dataset import Dataset
        test_ds = Dataset(shape='2D', test=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size)

    else:
        print('No such model!')
        exit(0)

    model.load_state_dict(torch.load(model_dir))
    return model, test_dl


def test_and_save_seg(model, test_dl, shape):
    model.eval()

    scores = np.array([])

    for step, (img, mask, _, name) in enumerate(test_dl):
        if torch.cuda.is_available():
            img = img.cuda()

        pred = model(img)

        # get numpy arrays
        pred = pred.cpu().detach()
        pred = np.squeeze(pred.numpy())
        pred = (pred > 0.75).astype(int)
        mask = mask.detach().numpy()

        # metrics
        if not scores.any():
            scores = metrics(mask, pred)
        else:
            scores += metrics(mask, pred)

        # save
        if shape == '3D':
            pred_nii = sitk.GetImageFromArray(pred.astype(float))
            output_dir = os.path.join('out', 'pred', '3D', name[0])
            sitk.WriteImage(pred_nii, output_dir)
        else:
            output_dir = os.path.join('out', 'pred', '2D', name[0])
            cv2.imwrite(output_dir, pred*255)

    # print scores
    scores /= len(test_dl)
    return scores


def test_and_save_cls(model, test_dl):
    model.eval()
    preds = []
    labels = []
    names = []

    for step, (img, _, label, name) in enumerate(test_dl):
        if torch.cuda.is_available():
            img = img.cuda()

        pred = model(img)

        # get numpy arrays
        pred = pred.cpu().detach()
        pred = np.squeeze(pred.numpy())
        pred = (pred > 0.75).astype(int)
        label = label.detach().numpy().reshape(-1).astype(int)

        names.extend(name)
        preds.extend(pred)
        labels.extend(label)

    # caculate metrics
    scores = metrics(np.array(labels), np.array(preds))

    # save
    df = pd.DataFrame(data=[names, preds, labels])
    df = df.T
    df.columns = ['name', 'pred', 'label']
    df.to_csv('./out/pred/cls/cls:%s.csv' % datetime.datetime.now())

    return scores


if __name__ == '__main__':

    models = ['seg_3D', 'seg', 'cls', 'multi']
    model, test_dl = get_model(models[1])
    scores = test_and_save_seg(model, test_dl, '2D')

    print(scores)



