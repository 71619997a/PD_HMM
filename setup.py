'''Run this first to create necessary CSV files.'''
from shutil import copy2 as cp
from datetime import datetime as dt
from CSVIO import *
from data import data_window, condense_data
def condense_data_file(in_file, out_file, num_condense, offset=0, num_data=3):
    '''Widen window of observation of data by joining together a number of windows.

    Arguments:
    in_file -- the input data file, as out_data.csv is formatted
    out_file -- the output data file
    num_condense -- the number of windows to condense
    '''
    list_ = read_csv(in_file)
    write_csv(out_file, condense_data(list_, num_condense, offset, num_data))

cp('data/activity_level.csv', '.')
cp('data/in_data.csv', '.')
cp('data/tremor_score.csv', '.')
indata = read_csv('data/in_data.csv')
outdata = map(lambda row : row[2:], indata)
write_csv('out_data.csv', outdata)
condense_data_file('out_data.csv', 'out_data_200ms.csv', 10)
condense_data_file('out_data.csv', 'out_data_1s.csv', 50)
condense_data_file('out_data.csv', 'out_data_5s.csv', 250)

start = dt(2015, 3, 24, 10, 27, 17, 500000)
end = dt(2015, 3, 24, 10, 44, 40, 500000)
training_window = data_window(read_csv('out_data.csv'),  start, end, 20)
write_csv('training_window.csv', training_window)

start2 = dt(2015, 3, 24, 10, 58, 32, 500000)
end2 = dt(2015, 3, 24, 11, 13, 50, 500000)
testing_window = data_window(read_csv('out_data.csv'), start2, end2, 20)
write_csv('testing_window.csv', testing_window)

condense_data_file('tremor_score.csv', 'tremor_30s.csv', 6, 1, 1)
