#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
import os
from preprocessor import file2frame,normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
#from featureExtraction import *
from featureExtraction import *

data = file2frame("../datas.csv","../qrss.csv")


normalize(data)

data['QSslope'] = data.apply(QSslope,axis=1)
data['diff'] = data.apply(diff,axis=1)
data['QSLen'] = data.apply(lambda x: np.subtract(x['S'],x['Q']),axis=1)
data = data.drop(['ECG'],axis=1)
labels = data['LABEL']
features = data.drop(['LABEL'],axis=1)

print("FEATURE:")
print(features.head)
print("LABEL:")
print(labels.head)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"acc: {int(accuracy_score(y_test,y_pred)*100)}%")
print(f"f1:  {int(f1_score(y_test,y_pred)*100)}%")

