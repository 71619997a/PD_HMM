"""This module has file io functions designed for reading from and writing to
CSV and other files."""
from ast import literal_eval

def read_csv(filename):
    """Reads a CSV formatted file and returns a 2D list of the data.

    Arguments:
    filename -- filename of CSV file.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    return map(lambda s: s.split(","), lines)


def write_csv(filename, list_):
    """Writes the contents of a 2D list into a CSV formatted file.

    Arguments:
    filename -- filename of CSV file.
    list -- 2D list of data to be written.
    """
    str_lines = []
    for row in list_:
        str_row = ','.join(str(el) for el in row)        
        str_lines.append(str_row.strip())
    text = "\n".join(str_lines)
    with open(filename, 'w') as f:
        f.write(text)

def write_object(filename, object):
    with open(filename, 'w') as f:
        f.write(repr(object))

def read_object(filename):
    with open(filename, 'r') as f:
        return literal_eval(f.read())