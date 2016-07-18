from random import random
from NormalEmissionHMM import NormalEmissionHMM
from CSVIO import *
def test0():
    transition = [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.15, 0.1, 0.75]]
    emission = [[1, 0.7], [3, 0.4], [5, 0.9]]
    initial = [1, 0, 0]
    hmm = NormalEmissionHMM(transition, emission, initial)
    print hmm

    prac_seq = [random() * 6 for i in range(20)]
    print(hmm.viterbi(prac_seq))

    train_seq = [random() * 6 for i in range(3000)]
    hmm.train(train_seq)
    print hmm

    prac_seq = [random() * 6 for i in range(20)]
    print(hmm.viterbi(prac_seq))

# Averages the data, 50ms is very small of a window
def condenseData(inFile, outFile, numCondense):
    ls = readCSV(inFile)
    ls = ls[1:] # first line is garbage
    ret = []
    for i in range(0, len(ls) - numCondense, numCondense):
        window = ls[i : i + numCondense]
        newData = [i * 0.05, 0, 0, 0]
        for row in window:
            newData[1] += row[3]
            newData[2] += row[4]
            newData[3] += row[5]
        newData[1] /= numCondense
        newData[2] /= numCondense
        newData[3] /= numCondense
        ret.append(newData)
    writeCSV(outFile, ret)
