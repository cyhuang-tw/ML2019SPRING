import sys
import numpy as np
import pandas as pd
import keras
import csv
from keras import Sequential
from keras import backend as K
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from keras.layers import Activation

def test(modelFile,testFile):
    df = pd.read_csv(testFile,encoding='big5',header=None)
    mean = np.load(modelFile + '_mean.npy')
    std = np.load(modelFile + '_std.npy')
    rawData = df.values
    rawData[rawData == 'NR'] = '0'
    rawData = rawData[:,2:]
    rawData = rawData.astype(np.float32)
    days = rawData.shape[0]//18
    num = 8
    data = np.zeros((days,num*9))
    for i in range(days):
        tmp = rawData[18*i:18*(i+1),:]
        tmp[15,:] = np.deg2rad(tmp[15,:])
        cosW = tmp[16,:] * np.cos(tmp[15,:])
        sinW = tmp[16,:] * np.sin(tmp[15,:])
        tmp[15,:] = cosW
        tmp[16,:] = sinW
        tmp = np.delete(tmp,(0,1,3,4,5,6,11,13,14,17),axis=0)
        data[i,:] = tmp.flatten()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not std[j] == 0 :
                data[i][j] = (data[i][j]- mean[j]) / std[j]
    model = load_model(modelFile + '.h5')
    answer = model.predict(data, batch_size=1)
    K.clear_session()
    return answer


def genFile(y_test,outFile):
    f = open(outFile,"w")
    w = csv.writer(f)
    title = ['id','value']
    w.writerow(title) 
    for i in range(240):
        content = ['id_'+str(i),y_test[i][0]]
        w.writerow(content)
    f.close()

def main(modelFile,testFile,outFile):
    K.clear_session()
    y_test = test(modelFile,testFile)
    genFile(y_test,outFile)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])