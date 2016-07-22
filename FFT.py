import numpy as np
from numpy.linalg import norm
from numpy import absolute as mag
from scipy.fftpack import fftn, fft
from scipy.signal import find_peaks_cwt 
import matplotlib.pyplot as plot
from data import group_data
from detect_peaks import detect_peaks

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