import sys
import numpy as np
import pandas as pd
import torch
import csv
from torch.utils.data import DataLoader
from util import ImgDataset, MyNet, read_file

def genFile(y_test, output_file):
    f = open(output_file, "w")
    w = csv.writer(f)
    title = ['id', 'label']
    w.writerow(title) 
    for i in range(y_test.shape[0]):
        content = [str(i), str(y_test[i])[1]]
        w.writerow(content)
    f.close()

def main(file_name, model_files, output_file):
    data, index = read_file(file_name, train=False, aug=False)
    dataset = ImgDataset(data, index)
    dataLoader = DataLoader(dataset, batch_size=512, shuffle=False)
    net_list = []
    for file in model_files:
        net = torch.load(file)
        net.eval()
        net_list.append(net)

    ans = np.zeros((0, 1),dtype=np.int)
    for batch_id, batch in enumerate(dataLoader):
        img, _ = batch
        output = torch.zeros([img.size(0), 7]).cuda()
        for net in net_list:
            output += net(img.cuda())
        label = np.argmax(output.cpu().data.numpy(), axis=1).reshape(-1, 1)
        ans = np.concatenate((ans, label), axis=0)
    genFile(ans, output_file)

if __name__ == '__main__':
    file_name = sys.argv[1]
    model_files = [sys.argv[2], sys.argv[3], sys.argv[4]]
    output_file = sys.argv[5]
    main(file_name, model_files, output_file)