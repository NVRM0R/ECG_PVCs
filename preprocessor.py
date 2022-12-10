#!/usr/bin/python3
# Performs first data preprocessing
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def file2frame(dataPath,labelsPath):
    ecg = pd.read_csv(dataPath,names=["I","II","III"])['I']
    qrs = pd.read_csv(labelsPath,names=["Q","S","form"])
    qrsDataLength = qrs.shape[0]
    #data's each element:
    # ( [QRS], S, form )
    data = []
    for i, row in qrs.iterrows():
        if(row['Q']>ecg.shape[0]):
            break
        if(i<qrsDataLength-1) and (i>0):
            if(row['form'] in ('V','S','B')):
                xStart = row['S']-100
                xStop = row['Q']+100
                Scoord = row['S']-xStart
                Qcoord = row['Q']-xStart
                signal = ecg[xStart:xStop].to_numpy()
                if(row['form']=='V'):
                    # 1 stands for PCV
                    label = 1
                elif(row['form']=='S'):
                    label = 0
                else:
                    label = 0
                data.append([signal,Scoord,Qcoord,label])
    return pd.DataFrame(data,columns=['ECG','S','Q','LABEL'])

def normalize(data):
    for i, row in data.iterrows():
        x = row['ECG']/np.max(row['ECG'])
        x = x-np.mean(x)
        x = savgol_filter(x,10,4)
        data.at[i,'ECG'] = x
