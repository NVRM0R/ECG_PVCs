


from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from slicer import calcAvgComplex1Mer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


ecg = pd.read_csv("dataS.csv",names=["I","II","III"])
qrs = pd.read_csv("qrsS.csv")


ecg_1 = ecg["I"]
ecg_2 = ecg["II"]
ecg_3 = ecg["III"]


# 1.extracting raw data to prepare features, sort raw data
ECGpathology = []
Wpathology = []

ECGok = []
Wok = []
Sform = []
Bform = []
Vform = []
skipped = 0
for x in range(qrs.shape[0]-1):
    inxStart = qrs['Q'][x]
    inxStop = qrs['Q'][x+1]
    ECGslice = ecg_1[inxStart:inxStop].to_numpy()
    avgSignal = np.average(ECGslice)
    ECGslice-=avgSignal
    padding = 300-ECGslice.shape[0]
    if(padding>0):
        ECGslice = np.pad(ECGslice, (0,padding), 'constant')
        inxDelta = inxStop-inxStart
    else:
        skipped+=1

    if(qrs['form'][x]=='V'):
        # pathology
        ECGpathology.append(ECGslice)
        Wpathology.append(inxDelta)
        Vform.append(ECGslice)
    elif (qrs['form'][x]=='S') or (qrs['form'][x]=='B'):
        # OK
        ECGok.append(ECGslice)
        Wok.append(inxDelta)
        if (qrs['form'][x]=='S'):
            Sform.append(ECGslice)
        elif (qrs['form'][x]=='B'):
            Bform.append(ECGslice)

print(f"colleted data: \nB={len(Bform)}\nS={len(Sform)}\nV={len(Vform)}\n\nskipped={skipped}")
# 2 analyzing features
# 2.1 analyzing Widths

#data = [Wpathology, Wok]
#bp = plt.boxplot(data)

# 2.2 RMSD
OkRmsdS = calcAvgComplex1Mer(ecg_1,qrs,'S')
OkRmsdB = calcAvgComplex1Mer(ecg_1,qrs,'B')
pathologyRmsdV = calcAvgComplex1Mer(ecg_1,qrs,'V')

RMSD_S = []
for x in Sform:
    #print(x)
    values = np.sqrt(((pathologyRmsdV - x) ** 2).mean())
    RMSD_S.append(values)

RMSD_V = []
for x in Vform:
    #print(x)
    values = np.sqrt(((pathologyRmsdV - x) ** 2).mean())
    RMSD_V.append(values)

RMSD_B = []
for x in Bform:
    #print(x)
    values = np.sqrt(((pathologyRmsdV - x) ** 2).mean())
    RMSD_B.append(values)


dataSetX = np.concatenate((RMSD_S,RMSD_B,RMSD_V))
dataSetY = np.concatenate((np.ones(len(RMSD_S)+len(RMSD_B)),np.zeros(len(RMSD_V))))

X_train, X_test, y_train, y_test = train_test_split(dataSetX, dataSetY, train_size=0.7, random_state=42)
print(X_train.shape)
exit()
model = LogisticRegression().fit(X_train,y_train)

print(classification_report(y_test, model.predict(X_test)))

