"""Driver module for PD_HMM."""
from random import random
from datetime import datetime as dt
import numpy as np
import CSVIO
from hmmlearn.hmm import GMMHMM, GaussianHMM
np.set_printoptions(threshold=np.inf)

def test1():
    """Tests GMMHMM class."""
    hmm = GMMHMM(3, 3)
    rand_train = lambda: [random() * 6, random() * 4 + 1, random() * 10 - 2]
    train_seq = [rand_train() for i in range(3000)]  # 3000 triplets
    hmm.fit(train_seq)
    print hmm

    prac_seq = [rand_train() for i in range(20)]
    print hmm.decode(prac_seq)

    score_seq = [rand_train() for i in range(5)]
    print hmm.score(score_seq)

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

def test4():
    # lists of data in these csvs
    x = [float(i) for i in column(read_csv('tremor_30s.csv'), 1)]
    y = [float(i) for i in column(read_csv('activity_level_fixed.csv'), 2)]
    print np.corrcoef(x, y)
    plot.scatter(x, y)
    plot.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), 'r-') # line of best fit
    plot.show()

def test5():
    activeset = column(read_csv('activity_level_fixed.csv'), 2) # only the activity data numbers
    numset = [float(i) for i in activeset]
    train, test = numset[:len(numset) / 2], numset[len(numset) / 2:]

    hmm = GMMHMM(3, 3, n_init=10000)
    hmm.fit(np.reshape(train, (-1, 1))) # required bc data has only 1 feature
    print 'train done'
    print hmm.decode(np.reshape(test, (-1, 1)))

def test6():
    tremorset = column(read_csv('tremor_score_fixed.csv'), 2) # only the tremor data numbers
    numset = [float(i) for i in tremorset]
    train, test = numset[:len(numset) / 2], numset[len(numset) / 2:]

    hmm = GMMHMM(3, 3, n_init=10000)
    hmm.fit(np.reshape(train, (-1, 1))) # required bc data has only 1 feature
    print 'train done'
    print hmm.decode(np.reshape(test, (-1, 1)))

def test7():
    pass

if __name__ == "__main__":
    print repr([[1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]])