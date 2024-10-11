import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torch.functional as F
import torch

import model_SHG as md

class SkipBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(SkipBlock, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), 
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1) 
        )

    def forward(self, x):
        return self.skip(x)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 4, 2, 1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_size, out_size, 1, 1, bias=False),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.1))
        if dropout:
            layers.append(nn.Dropout(dropout))
        side = [
            nn.Conv2d(in_size, out_size, 2, 2, bias=False),
        ]
        self.model = nn.Sequential(*layers)
        self.side = nn.Sequential(*side)

    def forward(self, x):
        x = self.model(x) + self.side(x)
        return x


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 3, 1, 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        side = [
            nn.Conv2d(in_size, out_size, 1, 1, bias=False),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        self.side = nn.Sequential(*side)

    def forward(self, x, skip_input):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.model(x) + self.side(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.skip1 = SkipBlock(64, 64)
        self.down2 = UNetDown(64, 128)
        self.skip2 = SkipBlock(128, 128)
        self.down3 = UNetDown(128, 256)
        self.skip3 = SkipBlock(256, 256)
        self.down4 = UNetDown(256, 512)
        self.skip4 = SkipBlock(512, 512)
        self.down5 = UNetDown(512, 512)
        self.skip5 = SkipBlock(512, 512)
        self.down6 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Conv2d(128, 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Sigmoid()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, self.skip5(d5))
        u2 = self.up2(u1, self.skip4(d4))
        u3 = self.up3(u2, self.skip3(d3))
        u4 = self.up4(u3, self.skip2(d2))
        u5 = self.up5(u4, self.skip1(d1))


        return self.final(u5)