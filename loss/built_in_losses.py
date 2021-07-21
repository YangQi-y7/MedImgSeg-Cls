# link: https://pytorch.org/docs/stable/nn.html#loss-functions
import torch.nn as nn

# some example of built-in loss functions in pytorch
L1 = nn.L1Loss()
MSE = nn.MSELoss()

CE = nn.CrossEntropyLoss()
BCE = nn.BCELoss()
BCE_L = nn.BCEWithLogitsLoss()


# in use:
def get_loss(pred, mask):
    loss = L1(pred, mask)
    return loss
