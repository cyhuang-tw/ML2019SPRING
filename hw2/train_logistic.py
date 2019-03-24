import sys
import numpy as np
import matplotlib.pyplot as plt
from util import *

plot = False

def crossEntropy(y_predict, y_label):
    return -np.dot(y_label, np.log(y_predict)) - np.dot((1 - y_label), np.log(1 - y_predict))

def gradient(X, y_label, w, b, lamda):
    y_predict = getProb(X, w, b)
    pred_error = y_label - y_predict
    wGrad = -np.mean(np.multiply(pred_error.T, X.T), 1) + lamda * w
    bGrad = -np.mean(pred_error)
    return wGrad, bGrad

def loss(y_predict, y_label, lamda, w):
    return crossEntropy(y_predict, y_label) + lamda * np.sum(np.square(w))

def shuffle(data, label):
    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    return data[randomize], label[randomize]
    
def train_dev_split(data, label, dev=0.25):
    idx = int(round(len(data) * (1 - dev)))
    return data[:idx], label[:idx], data[idx:], label[idx:]

def accuracy(y_predict, y_label):
    return np.sum(y_predict == y_label) / len(y_predict)

def train(x_train, y_train):
    dev_size = 0.15
    x_train, y_train, x_dev, y_dev = train_dev_split(x_train, y_train, dev=dev_size)

    w = np.zeros((x_train.shape[1],)) 
    b = np.zeros((1,))

    reg = False
    if reg:
        lamda = 0.001
    else:
        lamda = 0
    
    max_iter = 40
    batch_size = 32
    lr = 0.2
    num_train = len(y_train)
    num_dev = len(y_dev)
    step = 1

    loss_train = []
    loss_validation = []
    train_acc = []
    dev_acc = []
    
    for epoch in range(max_iter):
        x_train, y_train = shuffle(x_train, y_train)
        
        for idx in range(int(np.floor(len(y_train) / batch_size))):
            X = x_train[idx * batch_size:(idx + 1) * batch_size]
            Y = y_train[idx * batch_size:(idx + 1) * batch_size]

            wGrad, bGrad = gradient(X, Y, w, b, lamda)

            w = w - lr / np.sqrt(step) * wGrad
            b = b - lr / np.sqrt(step) * bGrad
            
            step += 1

        y_prob = getProb(x_train, w, b)
        y_predict = np.round(y_prob)
        train_acc.append(accuracy(y_predict, y_train))
        loss_train.append(loss(y_prob, y_train, lamda, w) / num_train)
        
        y_prob = getProb(x_dev, w, b)
        y_predict = np.round(y_prob)
        dev_acc.append(accuracy(y_predict, y_dev))
        loss_validation.append(loss(y_prob, y_dev, lamda, w) / num_dev)
    
    return w, b, loss_train, loss_validation, train_acc, dev_acc

def main(trainData,trainLabel,model):
    x_train = readFile(trainData)
    y_train = readFile(trainLabel).flatten()
    col = [0,1,3,4,5,7,10,12,25,26,27,28]
    x_train, mean, std = normalize(x_train, col=col)
    #x_train = x_train[:,:-42]
    #x_train = np.delete(x_train, (1), axis=1)
    w, b, loss_train, loss_validation, train_acc, dev_acc = train(x_train, y_train)
    np.save(model + '_w.npy', w)
    np.save(model + '_b.npy', b)
    np.save(model + '_mean.npy', mean)
    np.save(model + '_std.npy', std)
    if plot:
        plt.plot(loss_train)
        plt.plot(loss_validation)
        plt.legend(['train', 'dev'])
        plt.show()
        plt.plot(train_acc)
        plt.plot(dev_acc)
        plt.legend(['train', 'dev'])
        plt.show()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])