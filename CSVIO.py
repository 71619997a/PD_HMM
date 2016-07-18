def readCSV(filename):
    with open(filename, 'r') as f:
        lines = f.readLines()
    return map(lambda s: s.split(","), lines)

def writeCSV(filename,lst):
    lines = map(lambda r: ",".join(r), lst)
    text = "\n".join(lines)
    with open(filename, 'w') as f:
        f.write(text)
