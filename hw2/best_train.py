import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

def readFile(fileName):
    df = pd.read_csv(fileName)
    rawData = df.values
    rawData = rawData.astype(np.float)

    return rawData

def preProcess(data):
    data = data[:,:-42]
    data = np.delete(data, (1), axis=1)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    for i in range(data.shape[1]):
        if std[i] != 0:
            data[:,i] = (data[:,i] - mean[i]) / std[i]

    return data

def main(xFile, yFile, modelName):
    x_train = readFile(xFile)
    y_train = readFile(yFile)
    x_train = preProcess(x_train)
    y_train = y_train.flatten()
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    joblib.dump(model, modelName)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])