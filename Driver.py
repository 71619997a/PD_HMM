"""Driver module for PD_HMM."""
from random import random
from datetime import datetime as dt
import numpy as np
from GMMEmissionHMM import GMMEmissionHMM
from CSVIO import *
from FFT import *
from data import *

np.set_printoptions(threshold=np.inf)

def test0():
    """Tests GMMEmissionHMM class."""
    hmm = GMMEmissionHMM(3, 3)
    rand_train = lambda: [random() * 6, random() * 4 + 1, random() * 10 - 2]
    train_seq = [rand_train() for i in range(3000)]  # 3000 triplets
    hmm.train(train_seq)
    print hmm

    prac_seq = [rand_train() for i in range(20)]
    print hmm.viterbi(prac_seq)

    score_seq = [rand_train() for i in range(5)]
    print hmm.score(score_seq)

def test1():
    """states:
    0. rest
    1. standing
    2. walking
        normal
        counting
        narrow
    3. up stairs
    4. down stairs
    5. left finger to nose
    6. right finger to nose
    7. alternating RH movement
    8. alternating LH movement
    9. writing
    10. typing
    11. assembling
    12. drinking
    13. organize
    14. folding
    15. sitting
    """
    training_raw = read_csv('training_window.csv')
    train_data = [[float(row[1]), float(row[2]), float(row[3])] for row in training_raw]
    testing_raw = read_csv('testing_window.csv')
    test_data = [[float(row[1]), float(row[2]), float(row[3])] for row in testing_raw]
    hmm = GMMEmissionHMM(3, 3)
    hmm.train(train_data)
    train = hmm.viterbi(train_data)
    test = hmm.viterbi(test_data)
    print train[1], '\n' * 20, test[1]
    print 'training score:', train[0]
    print 'testing score:', test[0]

def test2():
    training_data = read_csv('training_window.csv')
    start = dt(2015, 3, 24, 10, 28, 20, 500000)
    end = dt(2015, 3, 24, 10, 28, 49, 500000)
    walking_data = data_window(training_data, start, end, 20) # 30 second walk
    graph_fftn(walking_data)

    start2 = dt(2015, 3, 24, 10, 27, 17, 500000)
    end2 = dt(2015, 3, 24, 10, 27, 47, 500000)
    standing_data = data_window(training_data, start2, end2, 20)
    graph_fftn(standing_data)

    start3 = dt(2015, 3, 24, 10, 30, 59, 500000)
    end3 = dt(2015, 3, 24, 10, 31, 12, 500000)
    upstairs_data = data_window(training_data, start3, end3, 20)
    graph_fftn(upstairs_data)

    start4 = dt(2015, 3, 24, 10, 31, 22, 500000)
    end4 = dt(2015, 3, 24, 10, 31, 45, 500000)
    downstairs_data = data_window(training_data, start4, end4, 20)
    graph_fftn(downstairs_data)

def test3():
    training_data = read_csv('training_window.csv')
    start = dt(2015, 3, 24, 10, 28, 20, 500000)
    end = dt(2015, 3, 24, 10, 28, 49, 500000)
    walking_data = data_window(training_data, start, end, 20) # 30 second walk
    freq = fftn_map(walking_data)
    size_dict = find_peaks(freq)
    print size_dict
    print 50. / len(freq)
if __name__ == "__main__":
    test3()
