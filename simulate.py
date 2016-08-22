import numpy as np
import matplotlib.pyplot as plot
from random import uniform
from hmmlearn.hmm import PeakHMM
from hmmlearnold.hmm import GaussianHMM
from scipy.interpolate import interp1d as interp
from FFT import i_fft_window
from time import clock

# gaussian hmm that generates indexes of peaks, widths, and heights
def peak_hmm(
        num_peaks=5,
        rest_prob=0.015,
        switch_prob=0.05,
        group_size=6000,
        height_mult=1,
        covar_mult=1,
        width_mult=1):
    rand_array = lambda min, max, num: [uniform(min, max) for _ in range(num)]
    means = []
    covars = []
    for _ in range(3):
        # index mean + cv
        state_means = rand_array(group_size / 100, group_size - 50, num_peaks)
        state_covars = rand_array(
            group_size * group_size / 1200 / 1200,
            group_size * group_size / 100 / 100, num_peaks)
        # height mean + cv
        state_means.extend(
            list(
                np.array(
                    rand_array(
                        12,
                        100,
                        num_peaks)) *
                height_mult))
        state_covars.extend(rand_array(1, 50, num_peaks))
        # width mean + cv
        state_means.extend(list(np.array(rand_array(
            group_size / 500, group_size / 75,
            num_peaks)) * width_mult))
        state_covars.extend(rand_array(
            group_size * group_size / 12000 / 12000,
            group_size * group_size / 1500 / 1500, num_peaks
        ))
        means.append(state_means * 3)
        covars.append(list(np.array(state_covars * 3) * covar_mult))
    hmm = GaussianHMM(n_components=3)
    hmm.startprob_ = np.array([0.3334, 0.3333, 0.3333])
    hmm.transmat_ = np.array([
        [1 - rest_prob, rest_prob, 0],
        [switch_prob * 5 / 16, 1 - switch_prob, switch_prob * 11 / 16],
        [0, switch_prob, 1 - switch_prob]
    ])
    # emissions: num peaks * num characteristics = 15, assume same for each dimension rn
    # index * 5, height * 5, width * 5, one for each dim
    hmm.means_ = means
    hmm.covars_ = covars
    return hmm


def regroup_peaksets(peak_sets):  # takes [[z1....z3x y1.....3x.....],]
    num_peaks = len(peak_sets[0]) / 9
    new = []
    group = lambda l, n: [l[:n], l[n:2 * n], l[2 * n:]]
    return [[group(j, num_peaks) for j in group(i, num_peaks * 3)]
            for i in peak_sets]
    # returns [[ [[z1 2 3 4 5],[z6 7 8 9 10],[z11 12 13 14 15]], [[y....]] ],
    # next]


# Peaks:
# [[[[z_peak_index_1,2,3,4,5],[z_peak_height_1,2,3,4,5],[z_peak_width_1,2,3,4,5]],[y....],[x....]],rest]
def simulate_acceleration_data(peaks, group_size=6000, base=2, noisiness=2):
    # i = clock()
    frequency_grouped = [
        create_peak_graph(
            peak_set,
            group_size,
            base,
            noisiness) for peak_set in peaks]
    # print clock() - i, "to group peaks of hmm data"
    triplets = i_fft_window(frequency_grouped)
    return triplets


# take [ [[z1 2 3 4 5], [...]], [[y...]] ]
def create_peak_graph(peak_set, group_size=6000, base=2, noisiness=2):
    graph_set = []
    for dim_peaks in peak_set:  # [[z1 z2 z3 z4 z5], [z...]]
        points = np.full(group_size, base, dtype=float)
        indices = dim_peaks[0]  # [z1 2 3 4 5]
        heights = dim_peaks[1]
        widths = dim_peaks[2]
        for i, ind in enumerate(indices):  # z1
            index = int(ind)
            if widths[i] <= 0 or index < 0 or index > group_size or heights[
                    i] <= 0:
                continue
            left_width = uniform(0, widths[i])
            right_width = widths[i] - left_width
            x = [index - left_width, index, index + right_width]
            y = [0, heights[i], 0]
            f = interp(x, y)
            for j in range(int(index - left_width) + 1,
                           int(index + right_width) + 1):  # need interp
                if(0 <= j < len(points)):
                    points[j] += f(j)
        # now we have to add noise
        noise = np.random.normal(0, noisiness, group_size)
        # for i in range(group_size):
        #     if(not i in indices):  # peaks already have noise
        #         points[i] = abs(points[i] + noise[i])
        #     else:
        #         points[i] = abs(points[i])
        # for i in range(group_size):
        #     points[i] = abs(points[i])
        points += noise
        graph_set.append(np.abs(points))
    return graph_set
