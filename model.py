import os
import numpy as np

import torch
import torch.nn as nn

from layer import *


class KeyPointModel(nn.Module):
    def __init__(self):
        super(KeyPointModel, self).__init__()
        self.keyPointModel = nn.Sequential(
            VGG(3, 64),
            VGG(64, 128),
            VGG(128, 256),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(1039 * 839 * 256, 512),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 6),
        )

    def forward(self, x):
        return keyPointModel(x)


class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.classificationModel = nn.Sequential(
            VGG(1, 32),
            VGG(32, 64),
            VGG(64, 128),
            VGG(128, 128),
            VGG(128, 256),
            VGG(256, 384),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(2304, 2048),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(2048, 2048),
            nn.ELU(inplace=True),
            nn.Linear(2048, 1),
        )

    def forward(self, x):
        return self.classificationModel(x)
