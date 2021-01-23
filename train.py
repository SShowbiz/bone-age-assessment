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


def train(train_continue=False):
    # 연산 프로세서 정의하기
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 이미지 저장, 체크포인트 모델 저장을 위한 디렉토리 생성하기
    checkpoint_dir = "./checkpoint/train"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 네트워크 정의하기
    classificationModel = ClassificationModel().to(device)

    # 네트워크 초기화하기
    init_weights(classificationModel)

    # cost function 정의하기
    MAELoss = nn.L1Loss().to(device)

    # Optimizer 설정하기
    optim = torch.optim.Adam(
        classificationModel.parameters(), lr=2e-5, betas=(0.5, 0.999))

    startEpoch = 1
    if (train_continue):
        classificationModel, optim, startEpoch = load(
            checkpoint_dir=checkpoint_dir, classificationModel=classificationModel, optim=optim)

    optim = torch.optim.Adam(
        classificationModel.parameters(), lr=2e-5, betas=(0.5, 0.999))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optim, step_size=5, gamma=0.5)

    # 데이터 로딩하기
    imageTransforms = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(size=(260, 200)), transforms.ToTensor()])
    dataset = CustomDataset(image_data_dir="./datasets/train",
                            label_data_dir="./labels/train/train.csv", transforms=imageTransforms)
    dataLoader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=8)

    # 이미지 저장을 위한 functions 설정하기
    def fn_tonumpy(x): return x.to(
        'cpu').detach().numpy().transpose(0, 2, 3, 1)

    def fn_denorm(x): return (x * 0.5) + 0.5
    cmap = None

    # training 변수 설정
    numEpoch = 50
    numBatch = len(dataLoader)

    for epoch in range(startEpoch, numEpoch + 1):
        # train start
        classificationModel.train()

        # loss array 선언
        lossMAE = []

        for batch, data in enumerate(dataLoader, 1):
            # forward path
            dataId = data['id'].to(device)
            image = data['image'].to(device)
            boneAge = data['boneage'].to(device).unsqueeze(1)
            gender = data['gender'].to(device).unsqueeze(1)
            output = classificationModel(image)
            # print(dataId, ' ', boneAge, ' ', gender)

            # backward path
            print('output', output)
            optim.zero_grad()
            loss = MAELoss(output, boneAge)

            loss.backward()
            optim.step()

            # calculate loss
            lossMAE += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | MAE LOSS %.4f " %
                  (epoch, numEpoch, batch, numBatch, np.mean(lossMAE)))

        if epoch % 5 == 0 or epoch == numEpoch:
            save(checkpoint_dir=checkpoint_dir,
                 classificationModel=classificationModel, optim=optim, epoch=epoch)

        scheduler.step()
