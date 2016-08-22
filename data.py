import CSVIO as io
from datetime import datetime as dt
from datetime import timedelta
from copy import deepcopy as copy
import numpy as np
from detect_peaks import detect_peaks


def datetime_object(string, micro=True):
    """Converts a formatted string into a DT object. Formatted like in_data.csv.

    Arguments:
    string -- the string to be converted.
    """
    form = '%Y-%m-%d %H:%M:%S'
    if(micro):
        form += '.%f'
        string += '000'  # millis -> micros
    return dt.strptime(string, form)  # +'000' is ms -> micro s


def datetime_string(object, micro=True):
    """Opposite of dt_object"""
    form = '%Y-%m-%d %H:%M:%S'
    if(micro):
        form += '.%f'
    unformatted = object.strftime(form)
    if(micro):
        return unformatted[:-3]  # trim last 3 0's
    return unformatted  # if no micros, it's fine


def group_data(data, group_size):
    groups = []
    for i in range(0, len(data), group_size):
        groups.append(data[i: i + group_size])
    groups.append(data[i:])  # add rest of data to end
    return groups


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
    # time in ms between these datetimes
    millis_to_start = start_delta.total_seconds() * 1000
    start_index = millis_to_start / resolution
    assert start_index == int(start_index)  # index has to be integer
    end_delta = end - first
    millis_to_end = end_delta.total_seconds() * 1000
    end_index = millis_to_end / resolution
    assert end_index == int(end_index)
    return data[int(start_index): int(end_index)]


def condense_data(list_, num_condense, offset=0, num_data=3):
    ret = []
    for i in range(0, len(list_) - num_condense, num_condense):
        window = list_[i: i + num_condense]
        new_data = [list_[i][offset]] + [0] * num_data
        for row in window:
            for j in range(1, num_data + 1):
                new_data[j] += float(row[j + offset])
        for j in range(1, num_data + 1):
            new_data[j] /= num_condense

        ret.append(new_data)
    return ret


def extract_data(filename):
    raw = io.read_csv('training_window.csv')
    return [[float(row[1]), float(row[2]), float(row[3])] for row in raw]


def vertical_slice(data, start, end):
    return data[:, start:end]


def column(data, col):  # special case of vert slice
    data[:, col]


def interpolate_gaps(data, offset=0, num_data=1, resolution=5000, micro=True):
    """Finds gaps in data based on timestamps, then interpolates through them

    Arguments:
    data -- the data
    offset -- index of the timestamp
    num_data -- number of data that has to be interpolated
    resolution -- normal amount of time between data points, ms
    """
    # 0. Make a list with only necessary info, and with DT objects.
    cut_data = vertical_slice(data, offset, offset + num_data + 1)
    dt_data = [[datetime_object(row[0], micro)] + row[1:]
               for row in cut_data]  # changes dt string to dt object

    # 1. Find gaps, make a list of gap start indexes
    gap_list = []
    resolution /= 1000.  # resolution is now in seconds
    times = column(dt_data, 0)  # list of just the datetimes
    # the comprehension must be performed to flatten the list of lists from
    # v_slice
    last = times[0]
    for i in range(1, len(times)):
        datetime_obj = times[i]
        delta = datetime_obj - last
        if delta.total_seconds() > resolution:  # gap at index of last
            gap_list.append(i - 1)
        last = copy(datetime_obj)  # set last to datetime_obj

    # 2. We have a list of gaps, now we have to use interp from np for each
    # point.
    for start in gap_list[::-1]:  # go backwards to avoid the indexes changing
        gap_delta = times[start + 1] - times[start]
        start_data = cut_data[start][1:]  # just accel data
        end_data = cut_data[start + 1][1:]
        gap_size = gap_delta.total_seconds()  # gap size is diff in seconds
        dt_iter = copy(times[start])
        xp = [0, gap_size]
        fps = zip(start_data, end_data)  # pairs of x, y, and z for interp
        add_data = []
        for i in np.linspace(
                resolution,
                gap_size,
                num=gap_size /
                resolution -
                1,
                endpoint=False):  # iterate through each missing point
            # goes res secl  onds ahead in time
            dt_iter += timedelta(seconds=resolution)
            interpolated_data = [np.interp(i, xp, fps[j])
                                 for j in range(num_data)]
            new_data = data[start][:offset] + [datetime_string(
                copy(dt_iter), micro)] + interpolated_data + data[start][offset + num_data + 1:]
            # above line takes the not data from start and just copies it
            add_data.append(new_data)
        # shoves the additional data between start, start + 1
        data[start + 1: start + 1] = add_data


def eliminate_zeroes(data, index=1):
    """Finds zeroes in column #index of data and removes those rows."""
    i = 0
    while i < len(data):
        if(float(data[i][index]) == 0):
            del data[i]
            i -= 1
        i += 1
