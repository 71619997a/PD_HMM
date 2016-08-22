"""Driver module for PD_HMM."""
import numpy as np
from CSVIO import *
from data import *
from FFT import *
from peaks import *
from simulate import *
from matching import *
import warnings
from hmmlearnold.hmm import GaussianHMM
from hmmlearn.hmm import PeakHMM, GMMHMM
from enum import Enum
from time import clock
from random import randint, shuffle
from scipy.stats import mode
from math import ceil

getindex = lambda i, d: i


def test4():  # shows correlation
    # lists of data in these csvs
    x = [float(i) for i in column(read_csv('tremor_30s.csv'), 1)]
    y = [float(i) for i in column(read_csv('activity_level_fixed.csv'), 2)]
    print np.corrcoef(x, y)
    plot.scatter(x, y)
    plot.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), 'r-')  # line of best fit
    plot.show()


def test8():
    train = vertical_slice(read_csv('training_window.csv'), 1, 4)
    test = vertical_slice(read_csv('testing_window.csv'), 1, 4)
    characteristics = [
        getindex,
        getheight,
        find_peak_endpoints,
        find_peak_average,
        find_peak_triangle_size]
    hmm_types = ['gauss', 'gmm']
    covar_types = ['full', 'diag', 'spherical', 'tied']
    for i in characteristics:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # supress hmm div 0 warns
            train_formatted = emissions_from_fft_peaks(
                train, 750, peak_characteristic=i)
            test_formatted = emissions_from_fft_peaks(
                test, 750, peak_characteristic=i)
            for j in hmm_types:
                for k in covar_types:
                    if j == 'gmm':
                        k = 'diag'
                    print 'testing with', i, '&', j, '&', k,
                    for it in range(10):
                        h = hmm_from_fft_peaks(
                            train_formatted, hmm_type=j, formatted=True, n_mix=3, covar_type=k)
                        try:
                            # bad test, but helps decide which hmm to use
                            score = h.score(test_formatted)
                        except:
                            score = 'fail'
                        print score,
                    print
                print


def test9():
    peak_set = [[[16.5, 78.3, 223.2, 256.5, 400.234, 650.23], [
        30.6, 65.3, 42.9, 21.12, 73.123, 53.22], [7.9, 10.4, 4.3, 11.12, 8.7, 13.3]]]
    g = create_peak_graph(peak_set)

def guess_seq_from_test_data(
        group_size=1500,
        num_groups=400,
        iterations=1200,
        num_peaks_gen=7,
        num_peaks=5,
        thresh=0.8,
        force_num_peaks=True,
        covar_mult=25,
        height_mult=1,
        width_mult=1,
        peak_class=False,
        creator=None,
        decoder=None,
        decoderother=None):
    np.seterr('ignore')
    scores = [0, 0]
    if creator is None: creator = peak_hmm(
        num_peaks=num_peaks_gen,
        group_size=group_size,
        height_mult=height_mult,
        covar_mult=covar_mult,
        width_mult=width_mult)
    if decoderother is None: decoderother = GaussianHMM(3, n_iter=1000)
    if decoder is None: decoder = PeakHMM(3, n_iter=1000)
    for iterer in range(iterations):
        # i = clock()
        _, train_data = simulate_data(
            creator, num_peaks_gen, 2, num_groups, group_size)
        seq, test_data = simulate_data(
            creator, num_peaks_gen, 2, num_groups, group_size)
        # print clock() - i, " to simulate data"

        def peak_char(i, d):
            return [i, find_height(i, d), find_peak_width(i, d)]

        # i = clock()
        train_emits = emissions_from_fft_peaks(
            train_data,
            group_size=group_size,
            peak_characteristic=peak_char,
            num_peaks=num_peaks,
            thresh=thresh,
            force_num_peaks=force_num_peaks,
            peak_class=peak_class)
        test_emits = emissions_from_fft_peaks(
            test_data,
            group_size=group_size,
            peak_characteristic=peak_char,
            num_peaks=num_peaks,
            thresh=thresh,
            force_num_peaks=force_num_peaks,
            peak_class=peak_class)
        decoder.fit(train_emits)
        decoderother.fit(train_emits)
        guess_seq = decoder.decode(test_emits)[1]
        guess_seq_2 = decoderother.decode(test_emits)[1]
        decoder.fit(test_emits)
        decoderother.fit(test_emits)
        bseq1 = best_seq(seq[:-1], guess_seq, 3)
        bestscore1 = diff_list(bseq1, seq)
        print iterer, ':', bestscore1, 'diffs, out of', len(
            guess_seq), 'elements'
        scores[0] += bestscore1
        bseq2 = best_seq(seq[:-1], guess_seq_2, 3)
        bestscore2 = diff_list(bseq2, seq)
        print 'other: ', iterer, ':', bestscore2, 'diffs, out of', len(
            guess_seq), 'elements'
        scores[1] += bestscore2
    return np.array(scores, dtype=float) / iterations  # , creator, decoderother, decoder, other_bseq, norm_bseq


def plot_guess_seq_by_changing_covar(
        sdevmult_max=10,
        data_points=20,
        sample_size=4,
        group_size=750,
        num_groups=400,
        iterations=3,
        num_peaks_gen=7,
        num_peaks=3,
        thresh=0.8,
        force_num_peaks=True,
        height_mult=1,
        width_mult=2):
    xr = np.linspace(0, sdevmult_max, data_points)
    y = [[], []]
    for sdev_mult in xr:
        scores = np.array([0, 0], dtype=float)
        for _ in range(sample_size):
            go = False
            while not go:
                go = True
                try:
                    scores += guess_seq_from_test_data(
                                group_size,
                                num_groups,
                                iterations,
                                num_peaks_gen,
                                num_peaks,
                                thresh,
                                force_num_peaks,
                                sdev_mult *
                                sdev_mult,
                                height_mult,
                                width_mult)
                except:
                    go = False  # try until true
        means = scores / sample_size
        y[0].append(means[0])
        y[1].append(means[1])
    plot.plot(xr, y[0], 'b-')
    plot.plot(xr, y[1], 'r-')
    plot.plot(xr, np.poly1d(np.polyfit(xr, y[0], 1))(xr), 'g:')
    plot.plot(xr, np.poly1d(np.polyfit(xr, y[1], 1))(xr), 'k:')
    correlation = np.corrcoef(xr, y[0])[0, 1]
    corr2 = np.corrcoef(xr, y[1])[0, 1]
    plot.title('Correlation1: ' + str(correlation) + '\nCorrelation2: ' + str(corr2))
    plot.savefig(
        'gseqplotboth_{}x{}x{}x{}x{}_{}'.format(
            group_size,
            num_groups,
            iterations,
            sample_size,
            data_points,
            sdevmult_max))
    plot.clf()


def test11():
    _, data = simulate_data(num_groups=1, group_size=6000)
    emissions_from_fft_peaks(data, group_size=6000)
    plot.show()


def test12():
    adict = read_object('activity_dict_fft')
    for k, v in adict.iteritems():
        plot.plot(v[2], 'r-', v[1], 'g-', v[0], 'b-')
        plot.title(k)
        plot.show()


def diff_list(l1, l2):
    diffs = 0
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            diffs += 1
    return diffs


def simulate_data(
        hmm=None,
        num_peaks=5,
        noisiness=2,
        num_groups=100,
        group_size=6000):
    if(hmm is None):
        hmm = peak_hmm(num_peaks, group_size=group_size)
    samples = hmm.sample(num_groups)
    np.set_printoptions(threshold=np.inf)
    peak_data = regroup_peaksets(samples[0])
    return samples[1], simulate_acceleration_data(
        peak_data, noisiness=noisiness, group_size=group_size)


def emissions_from_fft_peaks(
        training_data, group_size=6000, min_frequency=1. / 3, num_peaks=3, thresh=0.8, mpd=10,
        resolution=20, peak_characteristic=getindex, force_num_peaks=False, peak_class=False, use_peak_score=True):  # freq is default
    grouped = fft_window(training_data, group_size)[
        :-2]  # last groups = crap data
    peaks = grouped_peakify(grouped,
                            num_peaks,
                            thresh,
                            mpd,
                            force_num_peaks,
                            1000. / (resolution * group_size),
                            min_frequency,
                            peak_class)
    if not peak_class:
        # peaks is [>[[zppp],[yppp],[xppp]]<,repeat] >state<
        feed_data = []
        for group_index, peak_index_set in enumerate(peaks):  # [[zppp],...]
            freq_list = []
            for j in peak_index_set: # [zppp]
                for i in sorted(j):
                    freq_list.append(i)  # flatten the peak indexes
            char_list = []
            for i, index in enumerate(freq_list):
                try:
                    char_list.extend(
                        peak_characteristic(
                            index, grouped[group_index][
                                i / num_peaks]))
                except TypeError:
                    char_list.append(  # peak char returns a number
                        peak_characteristic(
                            index, grouped[group_index][
                                i / num_peaks]))
            feed_data.append(char_list)
        return feed_data
    else:
        zeropeak = null_peak()
        feed_data = []
        for peak_set in peaks:  # first peak doesn't have score
            set_feed_data = []
            for dim_peak_set in peak_set:
                dim_feed_data = []
                if use_peak_score:
                    dim_feed_data.append(dim_peak_set[0])
                else:
                    for peak in dim_peak_set[1]:
                        dim_feed_data.extend(
                            [peak[1].index, peak[1].height])
                set_feed_data.extend(dim_feed_data)
            feed_data.append(set_feed_data)
        # print feed_data
        return feed_data


def test_overfit(
        group_size=1500,
        num_groups=400,
        num_peaks_gen=7,
        num_peaks=2,
        thresh=0.8,
        force_num_peaks=True,
        covar_mult=4,
        height_mult=4,
        width_mult=2,
        peak_class=False):
    creator = peak_hmm(
        num_peaks=num_peaks_gen,
        group_size=group_size,
        height_mult=height_mult,
        covar_mult=covar_mult,
        width_mult=width_mult)
    s = []
    ts = []
    lls = []
    tlls = []
    decoder = GaussianHMM(3, n_iter=1000)
    tseq, train_data = simulate_data(
        creator, num_peaks_gen, 2, num_groups, group_size)
    seq, test_data = simulate_data(
        creator, num_peaks_gen, 2, num_groups, group_size)

    def peak_char(i, d):
            return [i, find_height(i, d)]

    train_emits = emissions_from_fft_peaks(
        train_data,
        group_size=group_size,
        peak_characteristic=peak_char,
        num_peaks=num_peaks,
        thresh=thresh,
        force_num_peaks=force_num_peaks,
        peak_class=peak_class)
    test_emits = emissions_from_fft_peaks(
        test_data,
        group_size=group_size,
        peak_characteristic=peak_char,
        num_peaks=num_peaks,
        thresh=thresh,
        force_num_peaks=force_num_peaks,
        peak_class=peak_class)
    for hmm in decoder.fit(train_emits):
        print 'asd'
        tll, tgseq = hmm.decode(train_emits)
        ll, gseq = hmm.decode(test_emits)
        lls.append(ll)
        tlls.append(tll)
        state_shifts = [
            {0: 0, 1: 1, 2: 2},
            {0: 0, 1: 2, 2: 1},
            {0: 1, 1: 0, 2: 2},
            {0: 1, 1: 2, 2: 0},
            {0: 2, 1: 1, 2: 0},
            {0: 2, 1: 0, 2: 1}
        ]  # i would make a function but its annoying for a simple result
        # print len(seq), len(guess_seq)
        bestscore = diff_list(seq, gseq)
        bestlist = gseq
        for i in state_shifts:
            shifted_gseq = [i[j] for j in gseq]
            score = diff_list(seq, shifted_gseq)
            # print score
            if score < bestscore:
                bestscore = score
                bestlist = shifted_gseq

        print 'test:', bestscore, 'diffs, out of', len(
            gseq), 'elements'
        s.append(bestscore)
        bestscore = diff_list(tseq, tgseq)
        bestlist = tgseq
        for i in state_shifts:
            shifted_gseq = [i[j] for j in tgseq]
            score = diff_list(seq, shifted_gseq)
            # print score
            if score < bestscore:
                bestscore = score
                bestlist = shifted_gseq

        print 'train:', bestscore, 'diffs, out of', len(
            gseq), 'elements'
        ts.append(bestscore)
    plot.plot(ts, 'r-', s, 'g-')
    plot.show()
    plot.plot(tlls, 'r-', lls, 'g-')
    plot.show()


# def guess_seq_w_peak_predict(group_size=1500,
#         num_groups=402,
#         num_peaks_gen=7,
#         num_peaks=3,
#         thresh=0.8,
#         force_num_peaks=True,
#         covar_mult=1,
#         height_mult=1,
#         width_mult=1,
#         peak_class=False):
#     generator = peak_hmm(num_peaks=num_peaks_gen,
#         group_size=group_size,
#         height_mult=height_mult,
#         covar_mult=covar_mult,
#         width_mult=width_mult)
#     _, creator, decoderother, decoder, other_bseq, norm_bseq =
#         guess_seq_from_test_data(
#             group_size, 
#             num_groups,
#             num_peaks_gen,
#             num_peaks,
#             thresh,
#             force_num_peaks,
#             covar_mult,
#             height_mult,
#             width_mult)
#     # norm_bseq is only one we use
#     for n in range(3):
#         i = randint(len(norm_bseq))
#         same = randint(len(norm_bseq))
#         diff = randint(len(norm_bseq))
#         while norm_bseq[i] != n:
#             i = randint(len(norm_bseq))
#         while norm_bseq[same] != n:
#             same = randint(len(norm_bseq))
#         while norm_bseq[diff] == n:
#             diff = randint(len(norm_bseq))



def testdumb(
        group_size=1500,
        num_groups=400,
        num_peaks_gen=7,
        num_peaks=2,
        thresh=0.8,
        force_num_peaks=True,
        covar_mult=4,
        height_mult=4,
        width_mult=2,
        peak_class=True):
    creator = peak_hmm(
        num_peaks=num_peaks_gen,
        group_size=group_size,
        height_mult=height_mult,
        covar_mult=covar_mult,
        width_mult=width_mult)
    decoder = GaussianHMM(3, n_iter=1000)
    tseq, train_data = simulate_data(
        creator, num_peaks_gen, 2, num_groups, group_size)
    seq, test_data = simulate_data(
        creator, num_peaks_gen, 2, num_groups, group_size)

    def peak_char(i, d):
            return [i, find_height(i, d)]

    train_emits = emissions_from_fft_peaks(
        train_data,
        group_size=group_size,
        peak_characteristic=peak_char,
        num_peaks=num_peaks,
        thresh=thresh,
        force_num_peaks=force_num_peaks,
        peak_class=peak_class)
    test_emits = emissions_from_fft_peaks(
        test_data,
        group_size=group_size,
        peak_characteristic=peak_char,
        num_peaks=num_peaks,
        thresh=thresh,
        force_num_peaks=force_num_peaks,
        peak_class=peak_class)
    decoder.fit(train_emits)
    p = decoder.fit(test_emits)
    gseq = [i.argmax() for i in p]
    state_shifts = [
        {0: 0, 1: 1, 2: 2},
        {0: 0, 1: 2, 2: 1},
        {0: 1, 1: 0, 2: 2},
        {0: 1, 1: 2, 2: 0},
        {0: 2, 1: 1, 2: 0},
        {0: 2, 1: 0, 2: 1}
    ]  # i would make a function but its annoying for a simple result
    # print len(seq), len(guess_seq)
    bestscore = diff_list(seq, gseq)
    bestlist = gseq
    for i in state_shifts:
        shifted_gseq = [i[j] for j in gseq]
        score = diff_list(seq, shifted_gseq)
        # print score
        if score < bestscore:
            bestscore = score
            bestlist = shifted_gseq

    print 'test:', bestscore, 'diffs, out of', len(
        gseq), 'elements'


def shuffle_annotated_data(filename):
    data_dict = read_object(filename)
    keys = data_dict.keys()
    shuffle(keys)
    ret = []
    lengths = []
    for i in keys:
        lengths.append(len(data_dict[i]))
        ret.extend(data_dict[i])
    return keys, lengths, ret


def expected_sequence(lengths, emission_length):
    seq = []
    scaled = np.asarray(lengths, dtype=float) / emission_length
    leftover = 0
    for i, length in enumerate(scaled):
        base = int(length)
        seq.append(base)
        leftover += length - base
        if leftover >= 0.5:
            seq[i] += 1
            leftover -= 1
    return seq


def lists_to_dict(keys, vals):
    ret = {}
    for i, key in enumerate(keys):
        ret[key] = vals[i]
    return ret


if __name__ == "__main__":
    while True:
        group_size=200
        n_comp=4
        """beginning transmat:
        -each row of n elements must add to 1
        -row n: n has a 6/7 chance
        -other elements have = prob
        if x is chance of other elems, x * (n-1) + 6/7 = 1
        x = 1 / 7(n-1)"""
        transmat = np.full((n_comp, n_comp), 1. / 7 / (n_comp - 1))
        np.fill_diagonal(transmat, 6. / 7)

        def peak_char(i, d):
            return [i, find_height(i, d), find_peak_width(i, d)]

        activities_t, leng_t, train = shuffle_annotated_data('activity_dict2')
        activities, leng, test = shuffle_annotated_data('activity_dict1')
        activities2, leng2, test2 = shuffle_annotated_data('activity_dict1')
        train_emits = emissions_from_fft_peaks(
            train, group_size=group_size, peak_characteristic=peak_char, num_peaks=1)
        test_emits = emissions_from_fft_peaks(
            test, group_size=group_size, peak_characteristic=peak_char, num_peaks=1)
        test2_emits = emissions_from_fft_peaks(
            test2, group_size=group_size, peak_characteristic=peak_char, num_peaks=1)

        tryer = 0
        d = False
        while(tryer < n_comp):
            hmm = PeakHMM(n_comp, n_iter=1000, init_params='smc', params='tmc')
            hmm.fit(train_emits)
            try: _, tmpseq = hmm.decode(train_emits)
            except: d = True; break
            tryer = len(np.unique(tmpseq))
        # make sure all states are used
        if d: continue
        def subroutine(activities, leng, emits):
            exp_seq = expected_sequence(leng, group_size)
            try: _, seq = hmm.decode(emits)
            except: return
            start = 0
            formatted_seq = []
            for i in exp_seq:
                formatted_seq.append(seq[start : start + i].tolist())
                start += i
            activity_dict = lists_to_dict(activities, formatted_seq)
            newactivities = activities[:]
            for activity in activities:
                if activity[-1] == 'i' and activity[:-1] in activity_dict:
                    activity_dict[activity[:-1]].extend(activity_dict[activity])
                    del activity_dict[activity]
                    newactivities.remove(activity)
            # to prevent removal while iterating
            return newactivities, activity_dict

        act_test, dict_test = subroutine(activities, leng, test_emits)
        _, dict_train = subroutine(activities_t, leng_t, train_emits)
        _, dict_test2 = subroutine(activities2, leng2, test2_emits)
        unknowns = 0
        for activity in act_test:
            print activity, ' ' * (32 - len(activity)),
            test_states = dict_test[activity]
            train_states = dict_train[activity]
            test2_states = dict_test2[activity]
            m = mode(test_states)
            m_t = mode(train_states)
            m2 = mode(test2_states)
            if m.count[0] > len(test_states) / 2.:
                print m.mode[0],
            else:
                # unknowns += 1
                print '?',
            # if m_t.count[0] > len(train_states) / 2.:
            #     print m_t.mode[0],
            # else:
            #     print '?',
            if m2.count[0] > len(test2_states) / 2.:
                print m2.mode[0],
            else:
                print '?',
            print ' ' if m.mode[0] == m2.mode[0] else 'x'
        # print '\n', unknowns
        print '\n'
