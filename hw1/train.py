import sys
import time
import numpy as np
import pandas as pd

def readFile(fileName):
    df = pd.read_csv(fileName,encoding='big5')
    rawData = df.values
    rawData[rawData == 'NR'] = '0'
    rawData = rawData[:,3:]
    rawData = rawData.astype(np.float32)
    data = np.zeros((18,24*20*12))
    for i in range(20*12):
        data[:,24*i:24*(i+1)] = rawData[18*i:18*(i+1),:]
    ans = data[9,:]
    num = data.shape[0]
    x = np.zeros((471*12,num*9))
    y = np.zeros((471*12,1))
    count = 0
    for i in range(data.shape[1]):
        if i % 480 <= 470:
            x[count,:] = data[:,i:i+9].flatten()
            y[count,:] = ans[i+9]
            count += 1
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not std[j] == 0:
                x[i,j] = (x[i,j] - mean[j]) / std[j]
    return x,y,mean,std

def train(fileName,modelName):
    #x_train,y_train = readFile(fileName)
    x_train,y_train,mean,std = readFile(fileName)
    x_train = np.concatenate((x_train,np.ones((x_train.shape[0],1))),axis=1)

    w = np.zeros((x_train.shape[1],1)) + 0.01
    #w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_train.transpose(),x_train)),x_train.transpose()),y_train)
    
    lr = 50
    epoch = 10000
    prev_grad = 0
    for i in range(epoch):
        y = np.dot(x_train,w)
        err = y_train - y
        grad = (-2)*np.dot(x_train.T,err)
        prev_grad += grad**2
        ada = np.sqrt(prev_grad)
        w -= lr * grad / ada
        if i % 1000 == 0 or i == epoch - 1:
            print(np.sqrt(np.average(np.square(err))))
    np.save(modelName + '.npy',w)
    np.save(modelName + '_mean.npy',mean)
    np.save(modelName + '_std.npy',std)

def main(fileName,modelName):
    tic = time.time()
    train(fileName,modelName)
    toc = time.time()
    #print(toc-tic)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])