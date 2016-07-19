"""This module has file io functions designed for reading from and writing to CSV files."""


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
    lines = map(lambda r: ",".join(r), list_)
    text = "\n".join(lines)
    with open(filename, 'w') as f:
        f.write(text)
