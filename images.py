#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from preprocessor import file2frame,normalize
#from featureExtraction import *

data = file2frame("../datas.csv","../qrss.csv")

# saving images path 
if not os.path.exists("images"):
    os.makedirs("images")


# data before preprocessing
plt.subplot(2,1,1)
for index,x in data[0:20].iterrows():
    plt.plot(x[0])
plt.title("Before preprocessing")

# Actual preprocessing
normalize(data)


# data after preprocessing
plt.subplot(2,1,2)
dataInfo = []
for index,x in data[0:20].iterrows():
    plt.plot(x[0])
plt.title("After preprocessing")
plt.tight_layout()

plt.savefig("images/preprocessing.png")

# first OK
plt.close()
plt.subplot(1,1,1)
plt.plot(data[data['LABEL']!=1].iloc[0]['ECG'])
plt.title("first OK")
plt.savefig("images/ok.png")

# first PVC
plt.close()
plt.subplot(1,1,1)
plt.plot(data[data['LABEL']==1].iloc[0]['ECG'])
plt.title("first PVC")
plt.savefig("images/pvc.png")





# save histogram of PVCs versus normal QRS
plt.close()
plt.subplot(1,1,1)
plt.hist(data['LABEL'])
plt.title("OK vs PVC")
plt.savefig("images/hist.png")

