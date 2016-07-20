import CSVIO as io
from datetime import datetime as dt

def datetime_object(string):
    """Converts a formatted string into a DT object. Formatted like in_data.csv.
    
    Arguments:
    string -- the string to be converted.
    """
    return dt.strptime(string + '000', '%Y-%m-%d %H:%M:%S.%f') # +'000' is ms -> micro s
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
    millis_to_start = start_delta.total_seconds() * 1000 # time in ms between these datetimes
    start_index = millis_to_start / resolution
    assert start_index == int(start_index) # index has to be integer
    end_delta = end - first
    millis_to_end = end_delta.total_seconds() * 1000
    end_index = millis_to_end / resolution
    assert end_index == int(end_index)
    return data[int(start_index) : int(end_index)]

def condense_data(list_, num_condense, offset=0, num_data=3):
    ret = []
    for i in range(0, len(list_) - num_condense, num_condense):
        window = list_[i : i + num_condense]
        new_data = [list_[i][offset]] + [0] * num_data
        for row in window:
            for j in range(1, num_data + 1):
                new_data[j] += float(row[j + offset])
        for j in range(1, num_data + 1):
            new_data[j] /= num_condense
            new_data[j] = str(new_data[j])

        ret.append(new_data)
    return ret

def extract_data(filename):
    raw = io.read_csv('training_window.csv')
    return [[float(row[1]), float(row[2]), float(row[3])] for row in raw]

def vertical_slice(data, start, end):
    ret = []
    for i in data:
        ret.append(i[start:end])
    return ret

def interpolate_gaps(data, offset=0, num_data=1, resolution=5000):
    """Finds gaps in data based on timestamps, then interpolates through them

    Arguments:
    data -- the data
    offset -- index of the timestamp
    num_data -- number of data that has to be interpolated
    resolution -- normal amount of time between data points, ms
    Returns fixed data set.
    """
    # 0. Make a list with only necessary info, and with DT objects.
    cut_data = vertical_slice(data, offset, offset + num_data + 1)
    dt_data = [[datetime_object(row[0])] + row[1:] for row in cut_data] # changes dt string to dt object

    # 1. Find gaps, make a list of gap start indexes
    gap_list = []
    resolution /= 1000. # resolution is now in seconds
    times = vertical_slice(dt_data, 0, 1) # list of just the datetimes
    last = times[0]
    for i in range(1,len(times)):
        datetime_obj = times[i]
        delta = datetime_obj - last
        if delta.total_seconds() > resolution: # gap at index of last
            gap_list.append(i - 1) 
        last = clone datetime_obj # set last to datetime_obj


