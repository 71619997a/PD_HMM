'''Run this first to create necessary CSV files.'''
from shutil import copy2 as cp
from datetime import datetime as dt
from ast import literal_eval
import numpy as np
from numpy import absolute as mag
from scipy.fftpack import fft
from detect_peaks import detect_peaks
from CSVIO import read_csv, write_csv, read_object, write_object
from data import data_window, condense_data, interpolate_gaps, eliminate_zeroes, vertical_slice, column, datetime_object
from FFT import fft_window


def data_rep(data):
    ret = '['
    for i in data:
        ret += '['
        for j in i:
            ret += str(j) + ';'
        ret = ret[:-1]
        ret += '];'
    ret = ret[:-1]
    return ret + ']'


def condense_data_file(in_file, out_file, num_condense, offset=0, num_data=3):
    '''Widen window of observation of data by joining together a number of windows.

    Arguments:
    in_file -- the input data file, as out_data.csv is formatted
    out_file -- the output data file
    num_condense -- the number of windows to condense
    '''
    list_ = read_csv(in_file)
    write_csv(out_file, condense_data(list_, num_condense, offset, num_data))


def setup_copydata():
    cp('data/activity_level.csv', '.')
    cp('data/in_data.csv', '.')
    cp('data/tremor_score.csv', '.')
    cp('data/activity_times1.csv', '.')
    cp('data/activity_times2.csv', '.')


def setup_annotateddata():
    indata = read_csv('data/in_data.csv')
    outdata = map(lambda row: row[2:], indata)
    write_csv('out_data.csv', outdata)
    condense_data_file('out_data.csv', 'out_data_200ms.csv', 10)
    condense_data_file('out_data.csv', 'out_data_1s.csv', 50)
    condense_data_file('out_data.csv', 'out_data_5s.csv', 250)


def setup_windows():
    start = dt(2015, 3, 24, 10, 27, 17, 500000)
    end = dt(2015, 3, 24, 10, 44, 40, 500000)
    training_window = data_window(read_csv('out_data.csv'), start, end, 20)
    write_csv('training_window.csv', training_window)

    start2 = dt(2015, 3, 24, 10, 58, 32, 500000)
    end2 = dt(2015, 3, 24, 11, 13, 50, 500000)
    testing_window = data_window(read_csv('out_data.csv'), start2, end2, 20)
    write_csv('testing_window.csv', testing_window)


def setup_tremor():
    tremor_data = read_csv('tremor_score.csv')
    eliminate_zeroes(tremor_data, 2)
    interpolate_gaps(tremor_data, 1, micro=False)
    write_csv('tremor_score_fixed.csv', tremor_data)
    condense_data_file('tremor_score_fixed.csv', 'tremor_30s.csv', 6, 1, 1)


def setup_activity():
    activity_data = read_csv('activity_level.csv')
    eliminate_zeroes(activity_data, 2)
    interpolate_gaps(activity_data, 1, micro=False, resolution=30000)
    write_csv('activity_level_fixed.csv', activity_data)


def setup_fft():
    l = read_csv('out_data.csv')
    out_20ms = vertical_slice(l, 1, 4)
    fft_data = fft_window(out_20ms, 1500)
    repped_data = [data_rep(i) for i in fft_data]
    times = column(l, 0)[::1500]
    time_fft_data = zip(times, repped_data)
    write_csv('fft_data_30s.csv', time_fft_data)

import matplotlib.pyplot as plot


def setup_peak():
    # 1500 data points, sampling rate 50hz
    # each point is 1/30 hz
    # 3-7 hz is around 90-210 datapts in
    l = read_csv('fft_data.csv')
    for i in l:
        i[1] = i[1].replace(';', ',')
    eval_list = [[dt, literal_eval(data)] for [dt, data] in l]
    top_peaks = []
    raw_peaks = []
    for i, group in enumerate(
            eval_list[:-1]):  # last freq set is cut off, kill it
        top_peaks.append([group[0]])
        raw_peaks.append([group[0]])
        for dim in group[1]:
            all_peaks = []
            mph = 10
            while len(all_peaks) < 3:
                all_peaks = detect_peaks(
                    dim[10: len(dim) / 2], mph=mph, mpd=10)
                mph -= 1
            index_fix = np.array(all_peaks) + 10
            sorted_peaks = sorted(
                index_fix,
                key=lambda index: -
                dim[index])  # - to sort fr top down
            raw_peaks[i].append(sorted_peaks)
            top = sorted_peaks[:3]
            top_peaks[i].append(top)

    write_csv('raw_peaks.csv', raw_peaks)
    write_csv('peaks_top3.csv', top_peaks)


def setup_activities(n):
    l = read_csv('activity_times'+n+'.csv')
    o = read_csv('out_data.csv')
    activity_windows = {}
    fft_activity = {}
    for row in l:
        start = datetime_object(row[0])
        end = datetime_object(row[1])
        activity = row[2].strip()
        while activity in activity_windows:
            activity += 'i'
        activity_windows[activity] = [
            list(row) for row in np.rot90(np.rot90(data_window(
                o, start, end, 20))[:3], 3).astype(float)]  # remove date
    write_object('activity_dict'+n, activity_windows)
    for act, data in activity_windows.iteritems():
        fft_activity[act] = [[mag(j) for j in fft(i)] for i in data]
    write_object('activity_dict'+n+'_fft', fft_activity)

setup_activities('1')
setup_activities('2')
