import numpy as np
from numpy.linalg import norm
from numpy import absolute as mag
from scipy.fftpack import fftn, fft
from scipy.signal import find_peaks_cwt 
import matplotlib.pyplot as plot
from data import group_data
from detect_peaks import detect_peaks

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

def fft_window(data, group_size): # DO NOT use column for data, USE VERTICAL SLICE
    """Makes windows of FFT'd data, end data is [[[z,z,z...],[y,y,y,y...],[x,x,x...]],...]"""
    # Step 0: group data
    grouped_data = group_data(data, group_size)

    # Step 1: FFT each group
    grouped_fft_data = []
    for i in grouped_data:
        rot = np.rot90(i)
        fftd = [fft(seq) for seq in rot]
        magnitudes = [[mag(j) for j in seq] for seq in fftd]
        grouped_fft_data.append(magnitudes) 
    return grouped_fft_data