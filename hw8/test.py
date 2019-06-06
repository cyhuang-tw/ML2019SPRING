import os
import sys
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from util import read_file, ImgDataset, ImgNet

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', default='./test.csv')
parser.add_argument('--output_file', default='./result.csv')
parser.add_argument('--model_file', default='./model.pth')

def gen_file(y_test, output_file):
    f = open(output_file, "w")
    w = csv.writer(f)
    title = ['id', 'label']
    w.writerow(title) 
    for i in range(y_test.shape[0]):
        content = [str(i), str(y_test[i])[1]]
        w.writerow(content)
    f.close()

def main(input_file, output_file, model_file):
    test_data, _ = read_file(input_file)
    test_set = ImgDataset(test_data)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=16)

    model = ImgNet().cuda()
    model.load_state_dict(torch.load(model_file))
    model.float()
    model.eval()

    ans = np.zeros((0, 1), dtype=np.int)

    for batch in test_loader:
        batch = batch.to('cuda')
        output = model(batch)
        label = np.argmax(output.cpu().data.numpy(), axis=1).reshape(-1, 1)
        ans = np.concatenate((ans, label), axis=0)
    gen_file(ans, output_file)

if __name__ == '__main__':
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    model_file = args.model_file
    main(input_file, output_file, model_file)