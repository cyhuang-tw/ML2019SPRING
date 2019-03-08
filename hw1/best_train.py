import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from keras import Sequential
from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.layers import Activation
import sys

def train(trainFile,modelFile):
    df = pd.read_csv(sys.argv[1],encoding='big5')
    rawData = df.values
    rawData[rawData == 'NR'] = '0'
    rawData = rawData[:,3:]
    rawData = rawData.astype(np.float32)
    data = np.zeros((18,24*20*12))
    for i in range(20*12):
        data[:,24*i:24*(i+1)] = rawData[18*i:18*(i+1),:]
    ans = data[9,:]
    data[15,:] = np.deg2rad(data[15,:])
    cosW = data[16,:] * np.cos(data[15,:])
    sinW = data[16,:] * np.sin(data[15,:])
    data[15,:] = cosW
    data[16,:] = sinW
    data = np.delete(data,(0,1,3,4,5,6,11,13,14,17),axis=0)

    num = data.shape[0]
    x_train = np.zeros((471*12,num*9))
    y_train = np.zeros((471*12,1))
    count = 0
    for i in range(data.shape[1]):
        if i % 480 <= 470:
            x_train[count,:] = data[:,i:i+9].flatten()
            y_train[count,:] = ans[i+9]
            count += 1
    mean = np.mean(x_train, axis = 0) 
    std = np.std(x_train, axis = 0)
    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[1]):
            if not std[j] == 0 :
                x_train[i][j] = (x_train[i][j]- mean[j]) / std[j]
    model = Sequential()
    model.add(Dense(42, input_shape=(8*9,), use_bias=True))
    model.add(Activation('relu'))
    model.add(Dense(1,use_bias=True))
    #model.compile(loss='mean_absolute_error',optimizer='Adadelta')
    model.compile(loss='mean_absolute_error',optimizer='Adam')
    history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.05, verbose=1)
    model.save(modelFile + '.h5')
    np.save(modelFile + '_mean.npy',mean)
    np.save(modelFile + '_std.npy',std)
    K.clear_session()

if __name__ == '__main__':
    K.clear_session()
    train(sys.argv[1],sys.argv[2])