from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

os.chdir("D:\!MOOC\Python_Directory\Data Analysis and Interpretation")

data = pd.read_csv('ool_pds.csv', sep=',', low_memory=False)
data.dtypes
data.describe()

data['ool'] = data['W1_F1'].astype('category') #factor variable

data['inc'] = pd.to_numeric(data['PPINCIMP'], errors='coerce') # qt var. - household income
data['soc'] = pd.to_numeric(data['W1_P2'], errors='coerce') # ft var. (can order) - social class
data['ethm'] = data['PPETHM'].astype('category')  # ft var. (can't order) - ethnic
data['sex'] = data['PPGENDER'].astype('category') # ft var. (binary) - gender
data['edu'] = pd.to_numeric(data['PPEDUCAT'], errors='coerce') # ft var. (can order) - education
data['age'] = pd.to_numeric(data['PPAGE'], errors='coerce') # qt var. - age
data['unemp'] = data['W1_P11'].astype('category') # ft var. (binary) - unemploy


s_data=data[['ool', 'inc', 'soc', 'ethm','sex','edu','age','unemp']]
s_data.dtypes
s_data.describe() #describe quantitative var.

s_data['ool'] = s_data['ool'].replace(-1, np.nan)
s_data['soc'] = s_data['soc'].replace(-1, np.nan)
s_data['unemp'] = s_data['unemp'].replace(-1, np.nan)

data_clean = s_data.dropna()

data_clean.dtypes
data_clean.describe()

recode1 = {1: 'positive', 2: 'neutral', 3: 'negative'}
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

print (data_clean.dtypes)
print (data_clean.describe())

predictors = data_clean[['inc', 'soc', 'ethm','sex','edu','age','unemp']]
targets = data_clean['ool']

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets,
                                                                test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape


print ('Training Set Frequency Table')
print (tar_train.value_counts(sort=False, normalize=True)) # -> always predict positive

print ('Test Set Frequency Table')
print (tar_test.value_counts(sort=False, normalize=True)) # 0.55 accuracy

classifier=DecisionTreeClassifier(max_leaf_nodes = 5)
classifier=classifier.fit(pred_train,tar_train)

prediction_train=classifier.predict(pred_train)
print ("Decision Tree - Training Set Result: Confusion Matrix & Accuracy")
print (sklearn.metrics.confusion_matrix(tar_train,prediction_train))
print (sklearn.metrics.accuracy_score(tar_train, prediction_train)) #model accuracy ~ 0.58

prediction_test=classifier.predict(pred_test)
print ("Decision Tree - Test Set Result: Confusion Matrix & Accuracy")
print (sklearn.metrics.confusion_matrix(tar_test,prediction_test))
print (sklearn.metrics.accuracy_score(tar_test, prediction_test)) #model accuracy = 0.56

from sklearn import tree
from io import StringIO
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out,
                    feature_names=pred_train.columns.values,
                    class_names = ['negative', 'neutral', 'positive'],filled=True, rounded=True)
                   
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
