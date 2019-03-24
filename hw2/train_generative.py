import sys
import numpy as np
import pandas as pd
from numpy.linalg import inv
from util import readFile, normalize

def train(x_train, y_train):
    class_0_id = []
    class_1_id = []
    for i in range(y_train.shape[0]):
        if y_train[i][0] == 0:
            class_0_id.append(i)
        else:
            class_1_id.append(i)

    class_0 = x_train[class_0_id]
    class_1 = x_train[class_1_id] 

    mean_0 = np.mean(class_0, axis=0)
    mean_1 = np.mean(class_1, axis=0)  

    n = class_0.shape[1]
    cov_0 = np.zeros((n,n))
    cov_1 = np.zeros((n,n))
        
    for i in range(class_0.shape[0]):
        cov_0 += np.dot(np.transpose([class_0[i] - mean_0]), [(class_0[i] - mean_0)]) / class_0.shape[0]

    for i in range(class_1.shape[0]):
        cov_1 += np.dot(np.transpose([class_1[i] - mean_1]), [(class_1[i] - mean_1)]) / class_1.shape[0]

    cov = (cov_0*class_0.shape[0] + cov_1*class_1.shape[0]) / (class_0.shape[0] + class_1.shape[0])
 
    w = np.transpose(((mean_0 - mean_1)).dot(inv(cov)) )
    b =  (- 0.5)* (mean_0).dot(inv(cov)).dot(mean_0) + 0.5 * (mean_1).dot(inv(cov)).dot(mean_1) + np.log(float(class_0.shape[0]) / class_1.shape[0]) 

    return w, b


def main(trainData, trainLabel, model):
    x_train = readFile(trainData)
    y_train = readFile(trainLabel)
    x_train, mean, std = normalize(x_train)
    w, b = train(x_train, y_train)
    np.save(model + '_w.npy', w)
    np.save(model + '_b.npy', b)
    np.save(model + '_mean.npy', mean)
    np.save(model + '_std.npy', std)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])