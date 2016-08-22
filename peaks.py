import numpy as np
from scipy.integrate import simps
from numpy import interp
from detect_peaks import detect_peaks
from functools import total_ordering
import peakutils
import copy
import matplotlib.pyplot as plot
from sklearn.utils.linear_assignment_ import linear_assignment


@total_ordering
class IndependentPeak:

    def __init__(self, index, height=None, width=None, data=None):
        self.height = height
        self.width = width
        self.index = index
        if data is not None:
            self.height = find_height(index, data)
            self.width = find_peak_width(index, data)

    def __eq__(self, other):
        try:
            return self.height == other.height and self.width == other.width and self.index == other.index
        except:
            if isinstance(other, int) or isinstance(other, float):
                return False
            else: 
                raise NotImplementedError("Can't compare IndependentPeak to type", str(type(other)))

    def __gt__(self, other):
        try:
            return self.height > other.height
        except:
            if isinstance(other, int) or isinstance(other, float):  #or type(other) == int or type(other) == float:
                return True  # for purpose of max with ints
            else: 
                raise NotImplementedError("Can't compare IndependentPeak to type", str(type(other)))

    def __deepcopy__(self, *args):
        return IndependentPeak(self.index, self.height, self.width)

    def match_cost(self, other, length):
        index_base_cost = abs(self.index - other.index)
        width_base_cost = abs(self.width - other.width)
        height_base_cost = abs(self.height - other.height)
        """for matching, index matters the most, let's say 55%
        then height matters second most, let's say 35%
        width matters least, let's say 10%
        """
        index_weighted_cost = index_base_cost * (250. / length)
        width_weighted_cost = width_base_cost * (1000. / length)
        height_weighted_cost = height_base_cost
        return 0.55 * index_weighted_cost + 0.35 * \
            height_weighted_cost + 0.1 * width_weighted_cost


def null_peak():
    return IndependentPeak(0, 0, 0)


def grouped_peakify(
        grouped_data,
        num_peaks=3,
        thresh=0.1,
        mpd=5,
        force_num_peaks=False,
        bin_size=1. / 30,
        min_frequency=1. / 3,
        peak_class=True,
        graph=False):
    """Grouped data in the format outputted by FFT.fft_window, i.e.
    [[[z,z,z...],[y,y,y,y...],[x,x,x...]],...]
    """
    c = 'bgr'
    # 0. Trim ends of data
    min_pt = int(min_frequency / bin_size) + 1  # ceiling
    max_pt = len(grouped_data[0][0]) / 2
    peakified = []
    original = None
    for i, dset in enumerate(grouped_data):  # [[zzz],[yyy],[xxx]]
        peakified.append([])  # to hold [ppp],[ppp],[ppp]
        for dimnum, dim in enumerate(dset):  # [zzz]
            if (not peak_class) or i == 0:
                # peakified[i].append(dim_peak_indices(
                #     dim, num_peaks, thresh, mpd, force_num_peaks,
                #     min_pt, max_pt))
                usable = dim[min_pt : max_pt]
                peakified[i].append([min_pt + j for j in alt_peak_find(usable, num_peaks, 2)] )
                if graph:
                    peakinds = peakified[i][dimnum]
                    x = np.arange(len(dim))
                    plot.plot(x, dim, c[dimnum] + '-')
                    y = [find_height(j, dim) for j in peakinds]
                    plot.plot([int(j) for j in peakinds], y, 'ko')

            else:  # dirty code alert
                prospect_indices = dim_peak_indices(
                    dim, num_peaks, thresh, mpd, True, min_pt, max_pt)
                prospect_peaks = [
                    IndependentPeak(
                        j, data=dim) for j in prospect_indices]
                orig_peaks = np.array(original[dimnum][1])[:, 1]
                score, matches = match_peaks(orig_peaks, prospect_peaks, len(dim))

                # pro_peak_copy = copy.deepcopy(prospect_peaks)
                # prev_peaks = np.array(peakified[i - 1][dimnum])[:, 1]
                # matches = np.asarray(match_peaks(prev_peaks, prospect_peaks, len(dim)))
                # nones = matches == -2
                # if np.any(nones):
                #     mindex = np.where(nones[:, 0])[0][0]
                # else:
                #     mindex = None
                # while mindex is not None:
                #     maxind = prospect_peaks.index(max(prospect_peaks))
                #     while maxind in matches[:, 1]:
                #         prospect_peaks[maxind] = -1
                #         maxind = prospect_peaks.index(max(prospect_peaks))
                #     matches[mindex] = np.asarray([mindex, maxind, -1])
                #     prospect_peaks[maxind] = -1
                #     nones = matches == -2
                #     if np.any(nones):
                #         mindex = np.where(nones[:, 0])[0][0]
                #     else:
                #         break
                ordered_peaks = [
                    [j[2], prospect_peaks[int(j[1])]] for j in matches]
                assert(len(ordered_peaks) == num_peaks)
                peakified[i].append([score, ordered_peaks])

            if peak_class and i == 0:  # turn it into independentpeaks
                peakified[0][dimnum] = [
                    0, [[-1, IndependentPeak(j, data=dim)] for j in peakified[0][dimnum]]]
        plot.show()
        if i == 0:
            original = peakified[0]
        # we now have [[ppp],[ppp],[ppp]]
    return peakified


def dim_peak_indices(
        dim,
        num_peaks,
        thresh,
        mpd,
        force_num_peaks,
        min_pt,
        max_pt):
    usable = np.array(dim[min_pt: max_pt])
    thr = thresh
    peaks = peakutils.indexes(usable, thr, mpd)
    while(len(peaks) < num_peaks and force_num_peaks):
        thr *= 0.9
        peaks = peakutils.indexes(usable, thr, mpd)
    x = np.arange(len(dim))
    # interp_peaks = peakutils.interpolate(
    #     x, dim, ind=peaks + min_pt, width=min_pt, func=peakutils.peak.centroid)
    sorted_top_peaks = sorted(
        peaks,
        key=lambda index: -
        find_height(
            index,
            dim))[
        :num_peaks]  # - to sort fr top down
    # [zzzz...] -> [peak,peak,peak]
    while len(sorted_top_peaks) < num_peaks:
        sorted_top_peaks.append(-1)
    return sorted_top_peaks

def alt_peak_find(data, num_peaks, thresh=10):
    datacopy = data.copy()
    indices = []
    while len(indices) < num_peaks:
        maxind = datacopy.argmax()
        indices.append(maxind)
        datacopy[max(maxind - thresh, 0) : maxind + thresh + 1] = 0
    return indices

# measure * height is what start + end must be less than
def find_peak_endpoints(index, data, measure=0.5):
    measure_point = find_height(index, data) * measure
    end_int = start_int = int(index)
    while data[start_int] > measure_point and start_int > 0:
        start_int -= 1
    while data[end_int] > measure_point and end_int < len(data) - 1:
        end_int += 1
    start = interp(measure_point, [data[start_int], data[
                   start_int + 1]], [start_int, start_int + 1])
    end = interp(measure_point, [data[end_int - 1],
                                 data[end_int]], [end_int, end_int + 1])
    return start, end


def find_peak_triangle_size(index, data, measure=0.5):
    start, end = find_peak_endpoints(index, data, measure)
    return find_peak_height(index, data) * (end - start)


def find_height(index, data):
    left = int(index)
    right = left + 1
    return np.interp(index, [left, right], [data[left], data[right]])


def find_peak_width(index, data, measure=0.3):
    a, b = find_peak_endpoints(index, data, measure)
    return b - a


def match_peaks(base_peaks, test_peaks, len_data):
    bp = np.array(base_peaks)
    tp = np.array(test_peaks)
    costs = np.zeros((len(bp), len(tp)))
    for i in range(len(base_peaks)):
        for j in range(len(test_peaks)):
            costs[i, j] = base_peaks[i].match_cost(test_peaks[j], len_data)
    lin = linear_assignment(costs)
    ret = []
    for pair in lin:
        cost = costs[pair[0], pair[1]]
        ret.append(np.append(pair, cost))
    score = np.sum(np.asarray(ret)[:, 2])
    return score, ret

if __name__ == "__main__":
    bp = []
    bp.append(IndependentPeak(100, 60, 10))
    bp.append(IndependentPeak(200, 35, 15))
    bp.append(IndependentPeak(600, 40, 7))
    bp.append(IndependentPeak(12, 40, 20))
    tp = []
    tp.append(IndependentPeak(150, 45, 8))  # should match to 1
    tp.append(IndependentPeak(120, 50, 12))  # should match to 0
    tp.append(IndependentPeak(500, 50, 12))  # should match to 2
    print match_peaks(bp, tp, 750)
