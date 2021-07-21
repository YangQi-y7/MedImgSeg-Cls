from net.Unet_2D import Unet
from net.pre_trained_models import ResNet18
import torch.nn as nn
import torch


class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.segment_net = Unet()
        self.classify_net = ResNet18(in_ch=2)
        self.connect_layer = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        pred_mask = self.segment_net(x)

        y = torch.cat((x, pred_mask), 1)

        pred_label = self.classify_net(y)

        return pred_mask, pred_label
