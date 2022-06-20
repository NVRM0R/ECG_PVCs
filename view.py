from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from slicer import calcAvgComplex1Mer

ecg = pd.read_csv("dataS.csv",names=["I","II","III"])
qrs = pd.read_csv("qrsS.csv")

start = 45000
stop =  48000
step = 1
x = range(start,stop,step)
ecg_1 = ecg["I"]
ecg_2 = ecg["II"]
ecg_3 = ecg["III"]
plt.plot(x,ecg_1[x])
print(qrs.shape)

for i in range(stop):
    if(qrs['S'][i]<start):
        continue
    if(qrs['S'][i]>stop):
        break
    if(qrs['form'][i] == 'V'):
        curCol = 'red'
    elif(qrs['form'][i] == 'S' or qrs['form'][i] == 'B'):
        curCol = 'green'
    else:
        curCol = 'yellow'
    plt.axvspan(qrs['Q'][i], qrs['S'][i], color=curCol, alpha=0.2)
plt.show()


#fig, axs = plt.subplots(3,1)
#axs[0].plot(ecg_1[x])
#axs[1].plot(ecg_2[x])
#axs[2].plot(ecg_3[x])