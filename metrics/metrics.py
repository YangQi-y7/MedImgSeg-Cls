import numpy as np
import glob
from hausdorff import hausdorff_distance
import os
from parameters import patches_dir
import SimpleITK as sitk


def dice(target, pred):
    smooth = 1e-5

    intersection = (pred * target).sum()
    union = (pred + target).sum()

    score = 2 * (intersection + smooth) / (union + smooth)
    return score


def recall(target, pred):
    smooth = 1e-5

    tp = (pred * target).sum()

    score = tp / (target.sum() + smooth)
    return score


def accuracy(target, pred):
    smooth = 1e-5

    correct = target.size - (target ^ pred).sum()
    total = target.size

    score = correct / (total + smooth)
    return score


def precision(target, pred):
    smooth = 1e-5

    tp = (pred * target).sum()

    score = tp / (pred.sum() + smooth)
    return score


def kap(target, pred):
    # Cohen Kappa Coefficient
    fa = target.size - (target ^ pred).sum()
    n = target.size
    fc = ((n - pred.sum())*(n - target.sum()) + target.sum()*pred.sum()) / n

    score = (fa - fc) / (n - fc)
    return score


def get_coordinates(matrix):
    coordinates = np.nonzero(matrix)
    ndim = len(coordinates)
    indexes = []
    for i in range(len(coordinates[0])):
        idx = []
        for j in range(ndim):
            idx.append(coordinates[j][i])
        indexes.append(idx)
    return np.array(indexes)


# 3D hausdorff， unoptimized
def hausdorff_95(target, pred):
    # get coordinates
    target_idx = get_coordinates(target)
    pred_idx = get_coordinates(pred)

    # comput distance
    dist_lst = []
    for idx in range(len(target_idx)):
        dist_min = 1000.0   # maximal distance
        for idx2 in range(len(pred_idx)):
            dist = np.linalg.norm(target_idx[idx] - pred_idx[idx2])
            if dist_min > dist:
                dist_min = dist
                if dist_min == 0:
                    continue
        dist_lst.append(dist_min)

    # return 95 hausdorff
    dist_lst.sort()
    index = int(0.95 * len(dist_lst))
    return dist_lst[index]


def metrics(target, pred):
    scores = []
    scores.append(dice(target, pred))
    scores.append(recall(target, pred))
    scores.append(precision(target, pred))
    scores.append(accuracy(target, pred))
    scores.append(kap(target, pred))
    # scores.append(hausdorff_95(target, pred))

    return np.array(scores)




# TP = intersection(pred, target).sum()
# TP + FP = pred.sum()
# TP + FN = target.sum()
# TP + FP + FN = (pred | target).sum()