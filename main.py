#!/usr/bin/python3

import matplotlib.pyplot as plt
from preprocessor import file2frame,normalize,exampleOne
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from os import path,makedirs
from shutil import rmtree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score,plot_confusion_matrix
from featureExtraction import *
from seaborn import heatmap
from sklearn.model_selection import GridSearchCV

data = file2frame("../data.csv","../qrs.csv")
Njobs=3
OUTPUT_DIR = 'images'
plt.style.use('dark_background')

if path.exists(OUTPUT_DIR):
    rmtree(OUTPUT_DIR)
makedirs(OUTPUT_DIR)

plt.clf()
fig = plt.gcf()
fig.set_dpi(1200)
fig.set_size_inches(18.5, 10.5, forward=True)
for i in range(5):
    plt.plot(data.ECG[i].tolist(),linewidth=3)
plt.savefig(path.join(OUTPUT_DIR,"original.png"))
exampleOne(data.ECG[0].tolist(),'images')
normalize(data)

plt.clf()
for i in range(5):
    plt.plot(data.ECG[i].tolist(),linewidth=3)
plt.savefig(path.join(OUTPUT_DIR,"normalized.png"))

plt.clf()
plt.axis('off')
plt.plot(data[data['LABEL']==1]['ECG'].tolist()[0],linewidth=3)
plt.plot(data[data['LABEL']==1]['ECG'].tolist()[1],linewidth=3)
plt.savefig(path.join(OUTPUT_DIR,"PVC.png"))


plt.clf()
plt.axis('off')
plt.plot(data[data['LABEL']==0]['ECG'].tolist()[0],linewidth=3)
plt.plot(data[data['LABEL']==0]['ECG'].tolist()[1],linewidth=3)
plt.savefig(path.join(OUTPUT_DIR,"OK.png"))

plt.clf()
plt.axis('off')
plt.plot(data[data['LABEL']==0]['ECG'].tolist()[0],linewidth=3)
plt.savefig(path.join(OUTPUT_DIR,"single.png"))

data['QSslope'] = data.apply(RSslope,axis=1)
data['diff'] = data.apply(diff,axis=1)
data['QSLen'] = data.apply(lambda x: np.subtract(x['S'],x['Q']),axis=1)
data['negative_R'] = data.apply(negative_r,axis=1)
data['energy'] = data.apply(energy,axis=1)

#data = data.drop(['ECG'],axis=1)
labels = data['LABEL']
features = data.drop(['LABEL','ECG'],axis=1)

print("FEATURE:")
print(features.head)
print("LABEL:")
print(labels.head)

data = data[:][0:17000]

plt.axis('on')
plt.figure(figsize=(8, 6), dpi=80)
dataHeat = data.drop(['LABEL'],axis=1)
dataplot = heatmap(dataHeat.corr(),cmap='viridis')
plt.savefig(path.join(OUTPUT_DIR,"heatmap.png"))


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

### LogReg
grid={
    'penalty':['l1','l2'],
    'class_weight':['balanced'],
    'solver':['liblinear','saga']
     }


logreg=GridSearchCV(LogisticRegression(random_state=0),grid).fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("\n\nlogreg scores:")
print('Accuracy  {:.4}'.format(accuracy_score(y_test,y_pred)))
print('F1        {:.4}'.format(f1_score(y_test,y_pred)))
print('Recall    {:.4}'.format(recall_score(y_test,y_pred)))
print('Precision {:.4}'.format(precision_score(y_test,y_pred)))
print('ROC-AUC   {:.4}'.format(roc_auc_score(y_test,y_pred)))

### KNN
grid={
    'algorithm':['ball_tree', 'kd_tree', 'brute'],
    'weights':['uniform', 'distance'],
    'n_neighbors':[2,3,5,8]
     }

knn=GridSearchCV(KNeighborsClassifier(n_jobs=Njobs),grid,n_jobs=Njobs).fit(X_train,y_train)
y_pred = knn.predict(X_test)

print("\n\nKNN scores:")
print('Accuracy  {:.4}'.format(accuracy_score(y_test,y_pred)))
print('F1        {:.4}'.format(f1_score(y_test,y_pred)))
print('Recall    {:.4}'.format(recall_score(y_test,y_pred)))
print('Precision {:.4}'.format(precision_score(y_test,y_pred)))
print('ROC-AUC   {:.4}'.format(roc_auc_score(y_test,y_pred)))

### XGBoost
grid = {
    'n_estimators':[10,100,1000] ,
    'booster':['gbtree','gblinear','dart']
}

xgb = GridSearchCV(xgb.XGBClassifier(n_jobs=Njobs),grid,n_jobs=Njobs).fit(X_train,y_train)
y_pred = xgb.predict(X_test)
print("\n\nXGBoost scores:")
print('Accuracy  {:.4}'.format(accuracy_score(y_test,y_pred)))
print('F1        {:.4}'.format(f1_score(y_test,y_pred)))
print('Recall    {:.4}'.format(recall_score(y_test,y_pred)))
print('Precision {:.4}'.format(precision_score(y_test,y_pred)))
print('ROC-AUC   {:.4}'.format(roc_auc_score(y_test,y_pred)))


plt.clf()
plot_confusion_matrix(knn,X_test,y_test)
plt.savefig(path.join(OUTPUT_DIR,"confusionMatrix.png"))