import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.misc
import scipy.ndimage
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

def read_file(file_name):
    df = pd.read_csv(file_name)
    df['feature'] = df['feature'].str.split(' ')
    raw_data = df['feature']
    data = np.empty((len(raw_data), 48 * 48), dtype=np.int)
    for i in range(len(raw_data)):
        data[i, :] = np.array(list(map(int, raw_data[i]))).astype(np.int)
    label = df.values[:, 0].astype(np.int).flatten()

    return data, label

def random_split(data, label, ratio=0.2):
    x = np.concatenate((label.reshape(-1, 1), data), axis=1)
    np.random.shuffle(x)
    index = int(len(x) * ratio)
    train_data = x[index:, 1:]
    train_label = x[index:, 0].flatten()
    val_data = x[:index, 1:]
    val_label = x[:index, 0].flatten()
    return train_data, train_label, val_data, val_label

def data_augment(feature, label):
    flip = np.empty(feature.shape)
    left = np.empty(feature.shape)
    right = np.empty(feature.shape)
    up = np.empty(feature.shape)
    down = np.empty(feature.shape)
    rotate_left = np.empty(feature.shape)
    rotate_right = np.empty(feature.shape)
    zoom = np.empty(feature.shape)

    start = time.time()
    for i in range(feature.shape[0]):
        img = feature[i,:].reshape(48, 48)
        flip[i,:] = (np.flip(img, axis=1)).flatten()
        left[i,:] = (np.pad(img, ((0,0), (0,5)), mode='constant')[:, 5:]).flatten()
        right[i,:] = (np.pad(img, ((0,0), (5,0)), mode='constant')[:, :-5]).flatten()
        up[i,:] = (np.pad(img, ((5,0), (0,0)), mode='constant')[5:, :]).flatten()
        down[i,:] = (np.pad(img, ((0,5), (0,0)), mode='constant')[:-5, :]).flatten()

        img_left = scipy.ndimage.rotate(img, 20)
        img_left = scipy.misc.imresize(img_left, (48, 48))
        rotate_left[i,:] = img_left.flatten()
        
        img_right = scipy.ndimage.rotate(img, -20)
        img_right = scipy.misc.imresize(img_right, (48, 48))
        rotate_right[i,:] = img_right.flatten()

        img_z = scipy.misc.imresize(img, (38, 38))
        img_z = np.pad(img_z, ((5,5), (5,5)), mode='edge')
        zoom[i,:] = img_z.flatten()

    aug_list = [feature, flip, left, right, up, down, rotate_left, rotate_right, zoom]

    data = np.empty((feature.shape[0] * len(aug_list), 48 * 48))

    for i in range(len(aug_list)):
        data[i * feature.shape[0]:(i + 1) * feature.shape[0],:] = aug_list[i]

    label = np.tile(label, len(aug_list))

    return data, label

class ImgDataset(Dataset):
    def __init__(self, data, label=None, transform=None):
        self.data = data.astype(np.uint8)
        self.label = label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index, :].reshape(48, 48)
        img = self.transform(img)
        if self.label is not None:
            return img, self.label[index]
        else:
            return img

class ImgNet(nn.Module):
    def __init__(self):
        super(ImgNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, groups=128, padding=1),
            nn.Conv2d(in_channels=128, out_channels=144, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(144),
            nn.MaxPool2d(2),
            nn.Dropout(0.35)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=144, kernel_size=3, groups=144),
            nn.Conv2d(in_channels=144, out_channels=169, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(169),
            nn.MaxPool2d(2),
            nn.Dropout(0.35)
            )
        self.fc1 = nn.Sequential(
            nn.Linear(676, 16),
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 7)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 676)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
