import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm3d(out_ch),

            nn.Conv3d(out_ch, out_ch, 3, 1, padding=1),
            nn.PReLU(),
            nn.BatchNorm3d(out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class DownConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 2, 2),
            nn.PReLU(),
            nn.BatchNorm3d(out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class UpConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 2, 2),
            nn.PReLU(),
            nn.BatchNorm3d(out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Unet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(Unet, self).__init__()

        ch = 16
        filters = [ch, ch * 2, ch * 4, ch * 8, ch * 16]
        
        self.in_conv = ConvBlock(in_ch, filters[0])

        self.conv1 = ConvBlock(filters[1], filters[1])
        self.conv2 = ConvBlock(filters[2], filters[2])
        self.conv3 = ConvBlock(filters[3], filters[3])
        self.conv4 = ConvBlock(filters[4], filters[4])

        self.down1 = DownConv(filters[0], filters[1])
        self.down2 = DownConv(filters[1], filters[2])
        self.down3 = DownConv(filters[2], filters[3])
        self.down4 = DownConv(filters[3], filters[4])

        self.up4 = UpConv(filters[4], filters[3])
        self.up3 = UpConv(filters[3], filters[2])
        self.up2 = UpConv(filters[2], filters[1])
        self.up1 = UpConv(filters[1], filters[0])

        self.up_conv4 = ConvBlock(filters[4], filters[3])
        self.up_conv3 = ConvBlock(filters[3], filters[2])
        self.up_conv2 = ConvBlock(filters[2], filters[1])
        self.up_conv1 = ConvBlock(filters[1], filters[0])

        self.out_conv = nn.Conv3d(filters[0], out_ch, 1, 1)

    def forward(self, x):

        in_ = self.in_conv(x)

        e1 = self.down1(in_)
        e1 = self.conv1(e1)

        e2 = self.down2(e1)
        e2 = self.conv2(e2)

        e3 = self.down3(e2)
        e3 = self.conv3(e3)

        e4 = self.down4(e3)
        e4 = self.conv4(e4)

        d4 = self.up4(e4)
        d4 = torch.cat((d4, e3), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, in_), dim=1)
        d1 = self.up_conv1(d1)

        out = self.out_conv(d1)
        torch.sigmoid_(out)

        return out


if __name__ == "__main__":
    x = torch.rand(1,1,32,128,128)
    
    net = Unet()

    out = net(x)

    print(out.shape, 'out')

