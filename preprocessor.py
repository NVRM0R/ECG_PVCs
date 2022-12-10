#!/usr/bin/python3
# Performs first data preprocessing
import pandas as pd
import numpy as np

def file2frame(dataPath,labelsPath):
    ecg = pd.read_csv(dataPath,names=["I","II","III"])
    qrs = pd.read_csv(labelsPath,names=["Q","S","form"])
    qrsDataLength = qrs.shape[0]
    #data's each element:
    # ( [QRS], S, form )
    data = []
    for i, row in qrs.iterrows():
        if(row['Q']>ecg.shape[0]):
            break
        if(i<qrsDataLength-1):
            xStart = row['Q']
            xStop = qrs['Q'][i+1]
            signal = ecg['I'][xStart:xStop].to_numpy()
            Scoord = row['S']-xStart
            data.append([signal,Scoord,row['form']]) 
    return data

def normalize(data):
    for x in data:
        x[0] = x[0]/np.max(x[0])
        x[0] -= np.mean(x[0])
