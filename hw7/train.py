import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn import cluster
from util import random_split, ImgDataset, ImgEncoder

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', default='./images')
parser.add_argument('--num_epoch', type=int, default=10)

def main(input_dir, num_epoch):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    train_list, val_list = random_split(os.listdir(input_dir))
    train_set = ImgDataset(input_dir, file_list=train_list)
    val_set = ImgDataset(input_dir, file_list=val_list)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=True)

    model = ImgEncoder((3, 32, 32)).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()

    best_loss = 1e5
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for i, batch in enumerate(train_loader):
            batch = batch.to('cuda')
            optimizer.zero_grad()
            _, output = model(batch)
            batch_loss = loss(output, batch)

            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
            progress = ('=' * int(float(i)/len(train_loader)*40)).ljust(40)
            print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, \
            (time.time() - epoch_start_time), progress), end='\r', flush=True)            

        train_loss = train_loss / len(train_loader)
        
        model.eval()
        for i, batch in enumerate(val_loader):
            batch = batch.to('cuda')
            _, output = model(batch)
            batch_loss = loss(output, batch)
            val_loss += batch_loss.item()

        val_loss = val_loss / len(val_loader)
        
        print('[%03d/%03d] %2.2f sec(s) Train Loss: %3.6f | Val Loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time() - epoch_start_time, \
            train_loss, val_loss))

        if val_loss < best_loss:
            torch.save(model, 'model.pth')
            best_loss = val_loss
            print('Model Saved!')

if __name__ == '__main__':
    args = parser.parse_args()
    input_dir = args.input_dir
    num_epoch = args.num_epoch
    main(input_dir, num_epoch)
