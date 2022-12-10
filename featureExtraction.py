


from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from slicer import calcAvgComplex1Mer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from math import sqrt,atan
# data:
# ECG
# Q
# S


def QSslope(row):
    Qx = row['Q']
    Sx = row['S']
    section = row['ECG'][Qx:Sx]
    Ymax = np.max(section)
    Xmax = np.argmax(section)

    Ymin = np.min(section)
    Xmin = np.argmin(section)
    return atan((Ymax-Ymin)/(Xmax-Xmin))

def diff(row):
    Qx = row['Q']
    Sx = row['S']
    section = row['ECG'][Qx:Sx]
    diff = np.diff(section,1,axis=0)
    signNum = np.sign(diff)
    count = np.sum(((np.roll(signNum, 1) - signNum) != 0))
    return count
