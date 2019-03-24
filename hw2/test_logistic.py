import sys
import numpy as np
import csv
from util import readFile, normalize, infer, genFile

def main(testFile, model, outFile):
    x_test = readFile(testFile)
    w = np.load(model + '_w.npy')
    b = np.load(model + '_b.npy')
    mean = np.load(model + '_mean.npy')
    std = np.load(model + '_std.npy')

    col = [0,1,3,4,5,7,10,12,25,26,27,28]
    x_test, _, _ = normalize(x_test, False, col, mean, std)
    #x_test = x_test[:,:-42]
    #x_test = np.delete(x_test, (1), axis=1)
    y_test = infer(x_test, w, b).reshape(-1,1)
    y_test = y_test.astype(np.int)
    genFile(y_test, outFile)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])