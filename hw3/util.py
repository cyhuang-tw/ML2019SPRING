import sys
import time
import numpy as np
import pandas as pd
import scipy.misc
import scipy.ndimage
import torch
import torch.nn as nn
from torch.utils.data import Dataset

outliers = [59,2059,2171,2809,3262,3931,4275,5082,5274,5439,5722,5881,6102,6458,6699,7172,7496,7527,7629,8030,8423,8737,8856,9026,9500,9673,9679,9797,10023,10423,10657,11244,11286,11295,11846,12289,12352,13148,13402,13697,13839,13988,14279,15144,15389,15655,15835,15838,15894,16540,17081,17486,18012,19238,19632,20222,20712,20817,21786,21817,22198,22314,22407,22927,23596,23894,24053,24333,24441,24891,25219,25603,25909,25967,26383,26860,26897,27021,27400,28153,28601]

def read_file(file, train=True, aug=False):
    print('Reading File')
    df = pd.read_csv(file)
    df['feature'] = df['feature'].str.split(' ')
    raw = df['feature']
    feature = np.empty((raw.shape[0], 48*48))
    for i in range(raw.shape[0]):
        feature[i,:] = np.array(list(map(int, raw[i])))
    label = df.values[:, 0].astype(np.int).reshape(-1, 1)
    if train:
        feature = np.delete(feature, outliers, axis=0)
        label = np.delete(label, outliers, axis=0)
        data = np.concatenate((feature, label), axis=1)
        np.random.seed(87)
        np.random.shuffle(data)
        feature = data[:,:-1]
        label = data[:,-1].reshape(-1,1)
        val_split = 0.2
        index = np.floor((1 - val_split) * feature.shape[0]).astype(np.int)
        train_feature, val_feature = np.split(feature, [index])
        train_label, val_label = np.split(label, [index])

        if aug:
            train_feature, train_label = data_augment(train_feature, train_label)

        return train_feature, train_label, val_feature, val_label
    else:
        return feature, label

def data_augment(feature, label):
    print('Data Augmentation')
    
    flip = np.empty(feature.shape)
    left = np.empty(feature.shape)
    right = np.empty(feature.shape)
    up = np.empty(feature.shape)
    down = np.empty(feature.shape)
    rotate_left = np.empty(feature.shape)
    rotate_right = np.empty(feature.shape)
    zoom = np.empty(feature.shape)
    #shear = np.empty(feature.shape)
    #shear_flip = np.empty(feature.shape)

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

        #shear_scale = np.array([[1, 0], [0, 1.5]])
        #img_shear = scipy.ndimage.interpolation.affine_transform(img, shear_scale)
        #shear[i,:] = img_shear.flatten()
        #shear_flip[i,:] = np.flip(img_shear, axis=1).flatten()
        
        progress = ('=' * int(float(i)/feature.shape[0]*40)).ljust(40)
        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (i + 1, feature.shape[0], \
        (time.time() - start), progress), end='\r', flush=True)
    print('\n')

    aug_list = [feature, flip, left, right, up, down, rotate_left, rotate_right, zoom]

    data = np.empty((feature.shape[0] * len(aug_list), 48 * 48))

    for i in range(len(aug_list)):
        data[i * feature.shape[0]:(i + 1) * feature.shape[0],:] = aug_list[i]

    label = np.tile(label, [len(aug_list), 1])
    print('Done.')

    return data, label


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class ImgDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature.astype(np.float32)
        self.label = label
        self.feature /= 255.0
        for i in range(self.feature.shape[0]):
            mean = np.mean(self.feature[i,:])
            std = np.std(self.feature[i,:])
            if std != 0:
                self.feature[i,:] = (self.feature[i,:] - mean) / std

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img = self.feature[idx, :]
        img = img.reshape(1, 48, 48)
        label = self.label[idx]
        return torch.tensor(img), torch.LongTensor(label)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.35)
            )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.35)
            )

        self.fc1 = nn.Sequential(
            nn.Linear(4608, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
            )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
            )

        self.fc3 = nn.Sequential(
            nn.Linear(512,7),
            #nn.Dropout(0.5)
            nn.Softmax(dim=1)
            )

        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)
        self.conv4.apply(gaussian_weights_init)

        self.fc1.apply(gaussian_weights_init)
        self.fc2.apply(gaussian_weights_init)
        self.fc3.apply(gaussian_weights_init)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x