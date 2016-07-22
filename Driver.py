"""Driver module for PD_HMM."""
from random import random
from datetime import datetime as dt
import numpy as np
import CSVIO
from hmmlearn.hmm import GMMHMM, GaussianHMM
np.set_printoptions(threshold=np.inf)

def test4(): # shows correlation
    # lists of data in these csvs
    x = [float(i) for i in column(read_csv('tremor_30s.csv'), 1)]
    y = [float(i) for i in column(read_csv('activity_level_fixed.csv'), 2)]
    print np.corrcoef(x, y)
    plot.scatter(x, y)
    plot.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), 'r-') # line of best fit
    plot.show()

def hmm_from_fft_peaks(training_data, group_size, min_frequency=1./3, num_peaks=3, init_mph=20, mpd=15, resolution=20, train_mode='freq', hmm_type='gmm'):
    # train_mode: freq, size, freqsize
    # hmm_type: gmm, gauss
    grouped = fft_window(training_data, group_size)
    peaks = grouped_peakify(grouped, num_peaks, init_mph, mpd, True, 1000. / (resolution * group_size), min_frequency)
    # peaks is [>[[ppp],[ppp],[ppp]]<,repeat] >state<
    feed_data = []
    for group_index, peak_index_set in enumerate(peaks): # [[ppp],...]
        emissions = []
        freq_list = [i for i in j for j in peak_index_set]
        if('freq' in train_mode):
            emissions.extend(freq_list)
        if('size' in train_mode):
            for dim_num, dim in enumerate(grouped[group_index]): # dim is one dimension of a group
                for i in peak_index_set[dim_num]: # i is the matching peak indices
                    emissions.append(dim[i]) # for every peak index i, find size at it, add to list
        feed_data.append(emissions)
    if(hmm_type == 'gmm'):
        hmm = GMMHMM(3, 3, n_iter=100)
    if(hmm_type == 'gauss'):
        hmm = GaussianHMM(3, n_iter=1000)
    hmm.fit(feed_data)
    return hmm

if __name__ == "__main__":
  