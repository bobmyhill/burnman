import os, sys, numpy as np, matplotlib.pyplot as plt

def listify_xy_file(filename):
    f=open(filename, 'r')
    data = []
    datastream = f.read()  # We need to open the file
    f.close()
    datalines = [ line.strip().split() for line in datastream.split('\n') if line.strip() ]
    for line in datalines:
        if line[0] != "%":
            data.append(map(float, line))

    data = np.array(zip(*data))
    return data
