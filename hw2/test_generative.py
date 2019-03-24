import sys
import numpy as np
from util import readFile, normalize, genFile

def sigmoid(data, w, b):
    arr = np.empty([data.shape[0],1],dtype=float)
    for i in range(data.shape[0]):
        z = data[i,:].dot(w) + b
        z *= (-1)
        arr[i][0] = 1 / (1 + np.exp(z))
    return np.clip(arr, 1e-8, 1-(1e-8))

def predict(data):
    ans = np.ones([data.shape[0],1])
    for i in range(data.shape[0]):
        if data[i] > 0.5:
            ans[i] = 0 
    return ans

def main(testData, model, outFile):
    x_test = readFile(testData)

    w = np.load(model + '_w.npy')
    b = np.load(model + '_b.npy')
    mean = np.load(model + '_mean.npy')
    std = np.load(model + '_std.npy')

    x_test, _, _ = normalize(x_test, train=False, col=None, mean=mean, std=std)
    y_prob = sigmoid(x_test, w, b)
    y_test = predict(y_prob).astype(np.int)
    genFile(y_test, outFile)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])