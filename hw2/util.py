import numpy as np
import pandas as pd
import csv

def readFile(fileName):
    df = pd.read_csv(fileName)
    data = df.values
    data = data.astype(np.float)
    return data

def normalize(data, train=True, col=None, mean=None, std=None):
    if train:
        if col == None:
            col = np.arange(data.shape[1])
        mean = (np.mean(data[:,col], axis=0)).reshape(1,-1)
        std = (np.std(data[:,col], axis=0)).reshape(1,-1)
    data[:,col] = np.divide(np.subtract(data[:,col], mean), std)
    return data, mean, std

def sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1-1e-6)

def getProb(x, w, b):
    return sigmoid(np.add(np.matmul(x, w), b))

def infer(x, w, b):
    return np.round(getProb(x, w, b))

def genFile(y_test, outFile):
    f = open(outFile, "w")
    w = csv.writer(f)
    title = ['id', 'label']
    w.writerow(title) 
    for i in range(y_test.shape[0]):
        content = [str(i+1), y_test[i][0]]
        w.writerow(content)
    f.close()