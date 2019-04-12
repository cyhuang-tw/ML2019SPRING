import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util import ImgDataset, MyNet, read_file

def main(train_file, num_epoch, model_file):
    train_feature, train_label, val_feature, val_label = read_file(train_file, train=True, aug=True)

    train_set = ImgDataset(train_feature, train_label)
    val_set = ImgDataset(val_feature, val_label)

    net = MyNet().cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    loss = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=True)

    best_acc = 0.0
    for epoch in range(num_epoch):
            epoch_start_time = time.time()
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0

            net.train()
            for i, batch in enumerate(train_loader):
                img, label = batch
                optimizer.zero_grad()
                output = net(img.cuda())
                batch_loss = loss(output, label.cuda().squeeze_())

                batch_loss.backward()
                optimizer.step()

                train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1).reshape(-1, 1) == label.numpy())
                train_loss += batch_loss.item()

                progress = ('=' * int(float(i)/len(train_loader)*40)).ljust(40)
                print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)
            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_set)

            net.eval()
            for i, batch in enumerate(val_loader):
                img, label = batch
                output = net(img.cuda())
                batch_loss = loss(output, label.cuda().squeeze_())

                val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1).reshape(-1, 1) == label.numpy())
                val_loss += batch_loss.item()
            val_loss = val_loss / len(val_loader)
            val_acc = val_acc / len(val_set)
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
                    (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                    train_acc, train_loss, val_acc, val_loss))

            if (val_acc > best_acc):
                '''
                with open('./acc.txt','w') as f:
                    f.write(str(epoch)+'\t'+str(val_acc)+'\n')
                '''
                torch.save(net, model_file)
                best_acc = val_acc
                print ('Model Saved!')

if __name__ == '__main__':
    train_file = sys.argv[1]
    num_epoch = int(sys.argv[2])
    model_file = sys.argv[3]
    main(train_file, num_epoch, model_file)

