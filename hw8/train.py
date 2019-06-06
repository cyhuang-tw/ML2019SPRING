import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from util import read_file, random_split, data_augment, ImgDataset, ImgNet

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='./train.csv')
parser.add_argument('--model_file', default='./model.pth')
parser.add_argument('--num_epoch', type=int, default=100)

def main(train_file, model_file, num_epoch):
    data, label = read_file(train_file)
    train_data, train_label, val_data, val_label = random_split(data, label)
    train_data, train_label = data_augment(train_data, train_label)
    
    train_set = ImgDataset(train_data, train_label, transform=None)
    val_set = ImgDataset(val_data, val_label)

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=True, num_workers=16)

    model = ImgNet().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(num_epoch):
            epoch_start_time = time.time()
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0

            model.train()
            for i, batch in enumerate(train_loader):
                img, label = batch
                img, label = img.to('cuda'), label.to('cuda')
                optimizer.zero_grad()
                output = model(img)
                batch_loss = loss(output, label)

                batch_loss.backward()
                optimizer.step()

                train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == label.cpu().numpy())
                train_loss += batch_loss.item()

                progress = ('=' * int(float(i)/len(train_loader)*40)).ljust(40)
                print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)
            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_set)

            model.eval()
            for i, batch in enumerate(val_loader):
                img, label = batch
                img, label = img.to('cuda'), label.to('cuda')
                output = model(img)
                batch_loss = loss(output, label)

                val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == label.cpu().numpy())
                val_loss += batch_loss.item()
            val_loss = val_loss / len(val_loader)
            val_acc = val_acc / len(val_set)
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
                    (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                    train_acc, train_loss, val_acc, val_loss))

            if (val_acc > best_acc):
                model.half()
                torch.save(model.state_dict(), model_file)
                model.float()
                best_acc = val_acc
                print ('Model Saved!')    



if __name__ == '__main__':
    args = parser.parse_args()
    train_file = args.train_file
    model_file = args.model_file
    num_epoch = args.num_epoch
    main(train_file, model_file, num_epoch)