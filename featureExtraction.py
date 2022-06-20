


from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ecg = pd.read_csv("dataS.csv",names=["I","II","III"])
qrs = pd.read_csv("qrsS.csv")


ecg_1 = ecg["I"]
ecg_2 = ecg["II"]
ecg_3 = ecg["III"]


# 1.extracting raw data to prepare features
ECGpathology = []
Wpathology = []

ECGok = []
Wok = []

for x in range(qrs.shape[0]-1):
    inxStart = qrs['Q'][x]
    inxStop = qrs['Q'][x+1]
    ECGslice = ecg[inxStart:inxStop].to_numpy()
    avgSignal = np.average(ECGslice)
    ECGslice-=avgSignal
    padding = 300-ECGslice.shape[0]
    if(padding>0):
        ECGslice = np.pad(ECGslice, (0,padding), 'constant')
        inxDelta = inxStop-inxStart

    if(qrs['form'][x]=='V'):
        # pathology
        ECGpathology.append(ECGslice)
        Wpathology.append(inxDelta)
    elif (qrs['form'][x]=='S') or (qrs['form'][x]=='B'):
        # OK
        ECGok.append(ECGslice)
        Wok.append(inxDelta)

# 2 analyzing features

data = [Wpathology, Wok]
bp = plt.boxplot(data)

# show plot
plt.savefig("WidthComp.png")
plt.show()