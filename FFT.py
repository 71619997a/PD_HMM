import numpy as np
import matplotlib.pyplot as plot
from data import group_data
from numpy import absolute as mag
from scipy.fftpack import rfft, irfft, fft, ifft
from random import uniform
from math import sqrt


def fft_window(data, group_size):  # DO NOT use column for data, USE VERTICAL SLICE
    """Makes windows of data then ffts it, end data is [[[z,z,z...],[y,y,y,y...],[x,x,x...]],...]"""
    # Step 0: group data
    grouped_data = group_data(data, group_size)

    # Step 1: FFT each group
    grouped_fft_data = []
    for i in grouped_data:
        rot = np.rot90(i)
        fftd = [abs(fft(seq) * (1. / group_size)) for seq in rot]
        grouped_fft_data.append(fftd)
    return grouped_fft_data


def i_fft_window(grouped_data):  # inverse of fft_window
    ifft_grouped = []
    for i, group in enumerate(
            grouped_data):  # [[z1,z2,z3,...], [y1,y2,y3...], [x...]]
        ifft_grouped.append([])
        for dim in group:
            ifft_grouped[i].append(irfft(dim))
    # put into triplets, then flatten
    group_triplets = [np.rot90(i, 3) for i in ifft_grouped]
    triplets = []
    for i in group_triplets:
        triplets.extend(i)
    return triplets
