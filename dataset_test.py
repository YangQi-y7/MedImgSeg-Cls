from dataset_prepare.dataset import Dataset
from torch.utils.data import DataLoader
from parameters import batch_size

# 2D dataset
train_ds = Dataset(shape='2D')
train_dl = DataLoader(train_ds, batch_size, shuffle=False)

print('total batch:', len(train_dl))
for step, (img, mask, label) in enumerate(train_dl):
    print(img.shape, mask.shape, label.shape)


# 3D dataset
train_ds = Dataset(shape='3D')
train_dl = DataLoader(train_ds, batch_size, shuffle=False)

print('total batch:', len(train_dl))
for step, (img, mask, label) in enumerate(train_dl):
    print(img.shape, mask.shape, label.shape)


