
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calcAvgComplex1Mer(ecg,qrsMapped,complexSym):
    i = 250
    ECGslices = []
    inxWidths = []
    build = 0
    maxInx = qrsMapped.shape[0]
    while (build<1000000) and (i<(maxInx-2)):
        i+=1
        if(qrs['form'][i] != complexSym):
            continue
        build+=1
        inxStart = qrsMapped['Q'][i]
        inxStop = qrsMapped['Q'][i+1]
        ECGslice = ecg[inxStart:inxStop].to_numpy()
        avgSignal = np.average(ECGslice)
        ECGslice-=avgSignal
        
        padding = 300-ECGslice.shape[0]

        if(padding>0):
            ECGslice = np.pad(ECGslice, (0,padding), 'constant')
            inxDelta = inxStop-inxStart
            inxWidths.append(inxDelta)
            ECGslices.append(ECGslice)
        else:
            continue
    ECGslices = np.array(ECGslices)
    avgECG = np.sum(ECGslices,axis=0)/ECGslices.shape[0]
    avgECG = avgECG/np.max(avgECG)
    return avgECG



#fig, axs = plt.subplots(3,1)
#axs[0].plot(ecg_1[x])
#axs[1].plot(ecg_2[x])
#axs[2].plot(ecg_3[x])