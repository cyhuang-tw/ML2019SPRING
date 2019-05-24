import os
import sys
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

def random_split(file_list, ratio=0.2):
    random.shuffle(file_list)
    index = int(len(file_list) * ratio)
    if ratio == 0.0:
        return file_list
    train_list = file_list[index:]
    val_list = file_list[:index]
    return train_list, val_list

class ImgDataset(Dataset):
    def __init__(self, path, file_list=None, transform=None):
        self.path = path
        if file_list is None:
            self.file_list = sorted(os.listdir(self.path))
        else:
            self.file_list = sorted(file_list)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
                ])
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, self.file_list[index])).convert('RGB')
        img = self.transform(img)
        img = img - 0.5
        return img

class ImgEncoder(nn.Module):
    def __init__(self, img_shape):
        super(ImgEncoder, self).__init__()
        self.channels, self.height, self.width = img_shape

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, return_indices=True)
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, return_indices=True)
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
            )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
            )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
            )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
            )

        self.unpool_1 = nn.MaxUnpool2d(2)

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
            )

        self.unpool_2 = nn.MaxUnpool2d(2)

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=self.channels, kernel_size=5),
            nn.Tanh()
            )

        self.encoder = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh()
            )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3072),
            nn.Tanh()
            )

    def forward(self, x):
        '''
        x = x.view(-1, 3072)
        x = self.encoder(x)
        y = self.decoder(x)
        y = y / 2.0
        y = y.view(-1, 3, 32, 32)
        '''
        x, idx_1 = self.conv1(x)
        x, idx_2 = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        y = self.deconv1(x)
        y = self.deconv2(y)
        y = self.unpool_1(y, idx_2)
        y = self.deconv3(y)
        y = self.unpool_2(y, idx_1)
        y = self.deconv4(y)
        y = y / 2.0
        return x, y