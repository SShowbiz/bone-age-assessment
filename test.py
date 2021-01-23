import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
import itertools
import matplotlib.pyplot as plt
import cv2

from torchvision import transforms
import torchvision


def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save(checkpoint_dir, classificationModel, optim, epoch):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save({'classificationModel': classificationModel.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (checkpoint_dir, epoch))


def load(checkpoint_dir, classificationModel, optim):
    if not os.path.exists(checkpoint_dir):
        epoch = 0
        return classificationModel, optim, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(checkpoint_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load(
        '%s/%s' % (checkpoint_dir, ckpt_lst[-1]), map_location=device)

    classificationModel.load_state_dict(dict_model['classificationModel'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return classificationModel, optim, epoch


def test():
    # 연산 프로세서 정의하기
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 불러오기를 위한 경로
    checkpoint_dir = "./checkpoint/train"

    # 네트워크 정의하기
    classificationModel = ClassificationModel().to(device)

    # 네트워크 초기화하기
    init_weights(classificationModel)

    # cost function 정의하기
    MAELoss = nn.L1Loss().to(device)

    # Optimizer 설정하기
    optim = torch.optim.Adam(
        classificationModel.parameters(), lr=2e-5, betas=(0.5, 0.999))

    # 모델 불러오기
    classificationModel, optim, startEpoch = load(
        checkpoint_dir=checkpoint_dir, classificationModel=classificationModel, optim=optim)

    # 데이터 로딩하기
    imageTransforms = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(size=(260, 200)), transforms.ToTensor()])
    dataset = CustomDataset(image_data_dir="./datasets/test",
                            label_data_dir="./labels/test/test.csv", transforms=imageTransforms)
    dataLoader = DataLoader(dataset, batch_size=1, num_workers=8)

    # training 변수 설정
    startEpoch = 1
    numEpoch = 1
    numBatch = len(dataLoader)

    with torch.no_grad():
        for epoch in range(startEpoch, numEpoch + 1):
            # test start
            classificationModel.eval()

            # loss array 선언
            lossMAE = []
            ageX = []
            ageY = []

            for batch, data in enumerate(dataLoader, 1):
                # forward path
                dataId = data['id'].to(device)
                image = data['image'].to(device)
                boneAge = data['boneage'].to(device).unsqueeze(1)
                gender = data['gender'].to(device).unsqueeze(1)
                output = classificationModel(image)
                loss = MAELoss(output, boneAge)

                # calculate loss
                lossMAE = loss.item()

                labelBoneAgeForYear = float(boneAge)/12
                predictBoneAgeForYear = float(output)/12

                ageX += [labelBoneAgeForYear]
                ageY += [predictBoneAgeForYear]

                print("TEST: BATCH %04d / %04d | LABEL: %5.2f | PREDICT: %5.2f | MAE LOSS %7.4f" %
                      (batch, numBatch, labelBoneAgeForYear, predictBoneAgeForYear, np.mean(lossMAE)))

    return ageX, ageY
