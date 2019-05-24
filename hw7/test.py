import os
import sys
import csv
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sklearn
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from util import ImgDataset, ImgEncoder

parser = argparse.ArgumentParser()

parser.add_argument('--model_file', default='./model.pth')
parser.add_argument('--input_dir', default='./images')
parser.add_argument('--test_file', default='./test_case.csv')
parser.add_argument('--output_file', default='./result.csv')

def read_file(file_name):
    df = pd.read_csv(file_name)
    data = df.values.astype(np.int)
    return data

def gen_file(y_test, output_file):
    f = open(output_file, "w")
    w = csv.writer(f)
    title = ['id', 'label']
    w.writerow(title) 
    for i in range(y_test.shape[0]):
        content = [str(i), str(y_test[i])]
        w.writerow(content)
    f.close()

def main(model_file, input_dir, test_file, output_file):
    encoder = torch.load(model_file)
    encoder.eval()

    dataset = ImgDataset(input_dir)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    img = np.zeros((len(dataset), 2048), dtype=np.float)
    count = 0
    for i, batch in enumerate(data_loader):
        batch = batch.to('cuda')
        output, _ = encoder(batch)
        output = output.cpu().data.numpy().reshape(-1, 2048)
        img[count:count + batch.size(0)] = output
        count += batch.size(0)
    pca_encoding = PCA(n_components=256, whiten=True, svd_solver="full", random_state=0).fit_transform(img)
    labels = KMeans(n_clusters=2, n_jobs=8, random_state=39).fit_predict(pca_encoding)

    test_data = read_file(test_file)
    answer = np.zeros(len(test_data), dtype=np.int)
    for i in range(len(test_data)):
        idx_1 = test_data[i, 1] - 1
        idx_2 = test_data[i, 2] - 1
        if labels[idx_1] == labels[idx_2]:
            answer[i] = 1
        else:
            answer[i] = 0
    gen_file(answer, output_file)

if __name__ == '__main__':
    args = parser.parse_args()
    model_file = args.model_file
    input_dir = args.input_dir
    test_file = args.test_file
    output_file = args.output_file
    main(model_file, input_dir, test_file, output_file)
