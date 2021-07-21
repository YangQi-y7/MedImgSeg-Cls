# link: https://pytorch.org/vision/stable/models.html
import torchvision.models as models
import torch.nn as nn
import torch

# some example of pre-trained models in pytorch
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
densenet = models.densenet161()
inception = models.inception_v3()
googlenet = models.googlenet()
mobilenet_v2 = models.mobilenet_v2()


# in use
class ResNet18(nn.Module):
    def __init__(self, in_ch):
        super(ResNet18, self).__init__()
        self.pre_layer = nn.Conv2d(in_ch, 3, kernel_size=3, stride=1, padding=1)
        self.pretained_resnet18 = models.resnet18(pretrained=True)
        self.post_layer = nn.Linear(1000, 1)

    def forward(self, x):
        y = self.pre_layer(x)
        z = self.pretained_resnet18(y)
        out = self.post_layer(z)

        torch.sigmoid_(out)
        return out
