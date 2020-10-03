from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

import sklearn.metrics
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier


os.chdir("D:\!MOOC\Python_Directory\Data Analysis and Interpretation")
data = pd.read_csv('ool_pds.csv', sep=',', low_memory=False)
data.dtypes
data.describe()

data['ool'] = data['W1_F1'].astype('category') #factor variable


data['inc'] = pd.to_numeric(data['PPINCIMP'], errors='coerce') 
data['soc'] = pd.to_numeric(data['W1_P2'], errors='coerce') 
data['ethm'] = data['PPETHM'].astype('category')  
data['sex'] = data['PPGENDER'].astype('category') 

data['edu'] = pd.to_numeric(data['PPEDUCAT'], errors='coerce') 
data['age'] = pd.to_numeric(data['PPAGE'], errors='coerce') 
data['unemp'] = data['W1_P11'].astype('category') 

s_data=data[['ool', 'inc', 'soc', 'ethm','sex','edu','age','unemp']]
s_data.dtypes
s_data.describe() #describe quantitative var.
s_data['ool'] = s_data['ool'].replace(-1, np.nan)
s_data['soc'] = s_data['soc'].replace(-1, np.nan)
s_data['unemp'] = s_data['unemp'].replace(-1, np.nan)

data_clean = s_data.dropna()

data_clean.dtypes

data_clean.describe()



recode1 = {1: 1, 2: 0, 3: -1}
print (data_clean["ool"].value_counts(sort=False)) #before recoding
data_clean['ool']= data_clean['ool'].map(recode1)
print (data_clean["ool"].value_counts(sort=False)) #after recoding

recode1 = {1:2500, 2:6250, 3:8750, 4:11250, 5:13750, 6: 17500, 7:22500, 8: 27500, 9: 32500,

          10:37500, 11: 45000, 12:55000, 13:67500, 14: 80000, 15: 92500, 16:112500,

          17: 137500, 18: 162500, 19: 200000}
print (data_clean["inc"].value_counts(sort=False)) #before recoding
data_clean['inc']= data_clean['inc'].map(recode1)
print (data_clean["inc"].value_counts(sort=False)) #after recoding
print (data_clean["soc"].value_counts(sort=False)) #before recoding
data_clean["soc"] = data_clean["soc"] -1
print (data_clean["soc"].value_counts(sort=False)) #after recoding
recode1 = {1: 0, 2: 1}
data_clean['sex']= data_clean['sex'].map(recode1)
data_clean['sex'] = data_clean['sex'].astype('category')
print (data_clean["sex"].value_counts(sort=False))
recode1 = {1: 1, 2: 0}

data_clean['unemp']= data_clean['unemp'].map(recode1)
data_clean['unemp'] = data_clean['unemp'].astype('category')
print (data_clean["unemp"].value_counts(sort=False))


def POSITIVE (row):

  if row['ool'] == 1 :

     return 1

  else :

      return 0

data_clean['pos'] = data_clean.apply (lambda row: POSITIVE (row),axis=1)

data_clean['pos'] = data_clean['pos'].astype('category')


data_clean.dtypes
data_clean.describe()
predictors = data_clean[['inc', 'soc', 'ethm','sex','edu','age','unemp']]

targets = data_clean['pos']

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets,

                                                                test_size=.4)

pred_train.shape

pred_test.shape

tar_train.shape

tar_test.shape

tar_train.describe() # 1 (positive) more often -> always predict positive
tar_test.describe() # 0.56


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)
predictions=classifier.predict(pred_test)


sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions) #0.57
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)

var_name = (pred_train.columns.tolist())
var_sig = (list(model.feature_importances_))
var_imp = DataFrame(columns=var_name)
var_imp.loc['Imp'] = [list(model.feature_importances_)[n] for n in range(7)]
var_imp[var_imp.ix[var_imp.last_valid_index()].argsort()[::-1]]

trees=range(25)
accuracy=np.zeros(25)
for idx in range(len(trees)):

  classifier=RandomForestClassifier(n_estimators=idx + 1)

  classifier=classifier.fit(pred_train,tar_train)

  predictions=classifier.predict(pred_test)

  accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)

plt.cla()

plt.plot(trees, accuracy)

plt.ylabel('Accuracy')

plt.xlabel('Number of Trees')

