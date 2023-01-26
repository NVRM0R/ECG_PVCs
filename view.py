#!/usr/bin/python3

from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from slicer import calcAvgComplex1Mer

ecg = pd.read_csv("../datas.csv",names=["I","II","III"])
qrs = pd.read_csv("../qrss.csv",names=["Q","S","form"])

start = 10000
stop =  15000
step = 1
x = range(start,stop,step)
ecg_1 = ecg["I"]
ecg_2 = ecg["II"]
ecg_3 = ecg["III"]
plt.gcf().set_dpi(1200)
plt.gcf().set_size_inches(18.5, 10.5, forward=True)
plt.style.use('dark_background')

plt.clf()
plt.subplot(3,1,1)
plt.plot(x,ecg_1[x])
plt.subplot(3,1,2)
plt.plot(x,ecg_2[x])
plt.subplot(3,1,3)
plt.plot(x,ecg_3[x])
plt.savefig("images/long.png")

#for i in range(stop):
#    if(qrs['S'][i]<start):
#        continue
#    if(qrs['S'][i]>stop):
#        break
#    if(qrs['form'][i] == 'V'):
#        curCol = 'red'
#    elif(qrs['form'][i] == 'S' or qrs['form'][i] == 'B'):
#        curCol = 'green'
#    else:
#        curCol = 'yellow'
#    plt.axvspan(qrs['Q'][i], qrs['Q'][i+1], color=curCol, alpha=0.2)
    
#plt.show()


#fig, axs = plt.subplots(3,1)
#axs[0].plot(ecg_1[x])
#axs[1].plot(ecg_2[x])
#axs[2].plot(ecg_3[x])