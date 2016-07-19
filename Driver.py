"""Driver module for PD_HMM."""
from random import random
from datetime import datetime as dt
import numpy as np
from NormalEmissionHMM import NormalEmissionHMM
from CSVIO import *


def test0():
    """Tests NormalEmissionHMM class."""
    transition = [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.15, 0.1, 0.75]]
    emission = [[1, 3, 5], [3, 5, 1], [5, 1, 3]]  # means, 3 emissions / state
    initial = [1, 0, 0]  # start in state 0
    hmm = NormalEmissionHMM(transition, emission, np.zeros((3, 3)), initial)
    print hmm
    rand_train = lambda: [random() * 6, random() * 4 + 1, random() * 10 - 2]
    train_seq = [rand_train() for i in range(3000)]  # 3000 triplets
    hmm.train(train_seq)
    print hmm

    prac_seq = [rand_train() for i in range(20)]
    print hmm.viterbi(prac_seq)

    score_seq = [rand_train() for i in range(5)]
    print hmm.score(score_seq)

# Averages the data, 50ms is very small of a window


def condense_data(in_file, out_file, num_condense):
    """Widen window of observation of data by joining together a number of windows.

    Arguments:
    in_file -- the input data file, as in_data.csv is formatted
    out_file -- the output data file
    num_condense -- the number of windows to condense
    Returns the array of out data.
    """
    list_ = read_csv(in_file)
    list_ = list_[1:]  # first line is garbage
    ret = []
    for i in range(0, len(list_) - num_condense, num_condense):
        window = list_[i: i + num_condense]
        new_data = [list_[i][2], 0, 0, 0]
        for row in window:
            new_data[1] += float(row[3])
            new_data[2] += float(row[4])
            new_data[3] += float(row[5])
        new_data[1] /= num_condense
        new_data[2] /= num_condense
        new_data[3] /= num_condense

        new_data[1] = str(new_data[1])
        new_data[2] = str(new_data[2])
        new_data[3] = str(new_data[3])

        ret.append(new_data)
    write_csv(out_file, ret)


def datetime_object(string):
    """Converts a formatted string into a DT object. Formatted like in_data.csv.
    
    Arguments:
    string -- the string to be converted.
    """
    return dt.strptime(string + '000', '%Y-%m-%d %H:%M:%S.%f') # +'000' is ms -> micro s
def data_window(data, start, end, resolution):
    """Extracts a window of data from a larger data list.

    Arguments:
    data -- the larger data list.
    start -- the datetime object that begins the window.
    end -- the datetime object that ends the window.
    resolution -- the length of one step in the data, in ms.
    Returns sublist of data sliced from start to end.
    """
    first = datetime_object(data[0][0])
    start_delta = start - first
    millis_to_start = start_delta.microseconds / 1000 # time in ms between these datetimes
    start_index = millis_to_start / resolution
    assert start_index == int(start_index) # index has to be integer
    end_delta = end - first
    millis_to_end= end_delta.microseconds / 1000
    end_index = millis_to_end / resolution
    assert end_index == int(end_index)
    return data[start_index : end_index]

start = dt(2015, 3, 24, 10, 27, 17, 500000)
end = dt(2015, 3, 24, 10, 44, 40, 500000)
annotated_window = data_window(read_csv('out_data_200ms.csv'),  start, end, 200)
write_csv('annotated_window.csv', annotated_window)


