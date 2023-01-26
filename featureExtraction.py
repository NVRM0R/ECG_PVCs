


from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import atan
# data:
# ECG
# Q
# S

# Slope between RS line and isoline
def RSslope(row):
    Qx = row['Q']
    Sx = row['S']
    section = row['ECG'][Qx:Sx]
    Ymax = np.max(section)
    Xmax = np.argmax(section)

    Ymin = np.min(section)
    Xmin = np.argmin(section)
    return atan((Ymax-Ymin)/(Xmax-Xmin))

# Fragmentation of the QRS complex
def diff(row):
    Qx = row['Q']
    Sx = row['S']
    section = row['ECG'][Qx:Sx]
    diff = np.diff(section,1,axis=0)
    signNum = np.sign(diff)
    count = np.sum(((np.roll(signNum, 1) - signNum) != 0))
    return count


def negative_r(row):
    Qx = row['Q']
    Sx = row['S']
    section = row['ECG'][Qx:Sx]
    Rmin = np.min(section)
    Rmax = np.max(section)
    Sy = row['ECG'][Sx]
    Qy = row['ECG'][Qx]
    negative = 0
    if (Rmin < 0) and ((Sy-Rmin) > 0.1) and ((Rmax-Qy) < 0.05):
        negative = 1
    return negative

def energy(row):
    value = np.cumsum(row['ECG'])[-1]
    return value