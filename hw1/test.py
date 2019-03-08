import sys
import numpy as np
import pandas as pd
import csv

def readFile(fileName,mean,std):
    df = pd.read_csv(fileName,encoding='big5',header=None)
    rawData = df.values
    rawData[rawData == 'NR'] = '0'
    rawData = rawData[:,2:]
    rawData = rawData.astype(np.float32)
    days = rawData.shape[0]//18
    num = 18
    data = np.zeros((days,num*9))
    for i in range(days):
        tmp = rawData[18*i:18*(i+1),:]
        data[i,:] = tmp.flatten()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not std[j] == 0:
                data[i,j] = (data[i,j] - mean[j]) / std[j]
    data = np.concatenate((data,np.ones((days,1))),axis=1)
    return data


def test(modelFile,inFile,outFile):
    w = np.load(modelFile + '.npy')
    mean = np.load(modelFile + '_mean.npy')
    std = np.load(modelFile + '_std.npy')
    x_test = readFile(inFile,mean,std)
    y_test = np.dot(x_test,w)
    y_test = np.abs(y_test)
    genFile(outFile,y_test)

def genFile(fileName,y_test):
    result = []
    for i in range(y_test.shape[0]):
        result.append(['id_'+str(i),float(y_test[i])])
    file = open(fileName,'w+')
    s = csv.writer(file,delimiter=',',lineterminator='\n')
    s.writerow(["id","value"])
    for i in range(y_test.shape[0]):
        s.writerow(result[i])
    file.close()

def main(modelFile,inFile,outFile):
    test(modelFile,inFile,outFile)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])