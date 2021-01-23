
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import torchvision
from skimage.transform import resize


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_data_dir, label_data_dir, transforms):
        self.image_data_dir = image_data_dir
        self.to_tensor = ToTensor()

        image_data = os.listdir(self.image_data_dir)
        image_data.sort()
        label_data = pd.read_csv(label_data_dir)
        self.image_data = image_data
        self.label_data = label_data
        self.id_data = label_data['id']
        self.boneage_data = label_data['boneage']
        self.gender_data = label_data['male']
        self.transforms = transforms

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        # channel 수 통일 (grayscale)
        image = cv2.imread(os.path.join(
            self.image_data_dir, self.image_data[idx]), flags=cv2.IMREAD_GRAYSCALE)
        boneage = self.boneage_data[idx]
        image = image[:, :, np.newaxis]
        image = image/255.0

        idValue = int(self.image_data[idx].split(".")[0])
        dataRow = self.label_data.loc[self.label_data['id'] == idValue]
        boneage = int(dataRow['boneage'])
        gender = int(dataRow['male'])

        image = image.astype(np.float32)
        image = self.transforms(image)
        data = {'id': idValue, 'image': image,
                'boneage': boneage, 'gender': gender}
        return data


class ToTensor(object):
    def __call__(self, data):
        data = torch.from_numpy(
            data.transpose((2, 0, 1)).astype(np.float32))
        return data


class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(
                self.shape[0], self.shape[1], self.shape[2]))

        return data
