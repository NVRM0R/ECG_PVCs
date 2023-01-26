#!/usr/bin/python3
# Performs first data preprocessing
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter,filtfilt,butter
import matplotlib.pyplot as plt
from os.path import join
# polynom degree
SavgolN = 10
# Savgol window
SavgolW = 4
# Butterworth's filter degree
ButterN  = 5
# Butterworth's lowpass cutoff freq
ButterWn = 0.3

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
            if(row['form'] == 'U'):
                continue
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
        #x = savgol_filter(x,SavgolN,4)
        B, A = butter(ButterN, ButterWn, output='ba')
        x = filtfilt(B,A, x)
        if(np.isnan(x).sum()>0):
            continue
        data.at[i,'ECG'] = x


def exampleOne(dataEcg,path):
    plt.clf()
    fig,ax = plt.subplots(4,1)
    fig.set_dpi(1200)
    fig.set_size_inches(17, 10, forward=True)

    ax[0].plot(dataEcg,linewidth=3)
    ax[0].set_title("Original Data")
    
    
    x = dataEcg/np.max(dataEcg)
    x = x-np.mean(x)
    ax[1].plot(x,linewidth=3)
    ax[1].set_title("Normalized")
    
    B, A = butter(ButterN, ButterWn, output='ba')
    x = filtfilt(B,A, x)
    ax[2].plot(x,linewidth=3)
    ax[2].set_title("Butterworth's LPF")

    x = savgol_filter(x,SavgolN,SavgolW)
    ax[3].set_title("Savitzky–Golay Filtered")
    ax[3].plot(x,linewidth=3)
    plt.tight_layout()
    plt.savefig(join(path,"prePipeline.png"))