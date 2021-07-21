import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."

        intersection = (predict * target).sum()
        union = (predict + target).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score
