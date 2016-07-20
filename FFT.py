import numpy as np
from numpy.linalg import norm
from numpy import absolute as mag
from scipy.fftpack import fftn
import matplotlib.pyplot as plot

def graph_fftn(data):
    trimmed_date = np.rot90(np.rot90(data)[:3], 3) # rotate, kill date, rerotate
    frequencies = fftn(trimmed_date)
    magnitudes = map(lambda triplet : [mag(i) for i in triplet], frequencies)
    freq_rot = np.rot90(magnitudes)

    N = len(freq_rot[0])
    bin_size = 50. / N
    xAxis = np.arange(0, 50, bin_size)
    plot.plot(xAxis, np.array(freq_rot[2]), 'r-')
    plot.plot(xAxis, np.array(freq_rot[1]), 'g-')
    plot.plot(xAxis, np.array(freq_rot[0]), 'b-')
    plot.plot(xAxis, [norm(t) for t in magnitudes], 'k:')
    plot.xlim(0.1, 49)
    plot.show()

def fftn_map(data):
    fft_list = []
    trimmed_date = np.rot90(np.rot90(data)[:3], 3) # rotate, kill date, rerotate
    frequencies = fftn(trimmed_date)
    magnitudes = map(lambda triplet : [mag(i) for i in triplet], frequencies)
    N = len(magnitudes)
    bin_size = 50. / N
    xAxis = np.arange(0, 50, bin_size)
    for i in range(N):
        fft_list.append([bin_size * i, norm(magnitudes[i])])
        fft_list[i] += magnitudes[i]
    return fft_list[1:]


def find_peaks(frequencies, starting_threshold=50, base=20, num_peaks=6):
    # frequencies should be in frequency space (i.e. fftn_map'd)
    size_dictionary = {} # dict of size:frequency
    i = -1
    while i < len(frequencies) / 2:
        i += 1
        pt = frequencies[i]
        if pt[1] > starting_threshold:
            print 1 
            start = i
            while frequencies[start][1] > base: start -= 1
            print 2
            end = i
            while frequencies[end][1] > base: end += 1
            print 3
            magnitudes = zip(*(frequencies[start:end]))[1]
            size = np.sum(magnitudes) 
            size_dictionary[size] = magnitudes.index(max(magnitudes)) + start
            # start is the offest of list magnitudes relative to frequencies
            i = end
    sort = sorted(size_dictionary)
    key_list = sort[-num_peaks:]
    ret = {}
    for i in key_list:
        ret[i] = size_dictionary[i]
    return ret