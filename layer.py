import os
import numpy as np

import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

    def forward(self, x):
        return self.vgg(x)
