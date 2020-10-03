Course 4 week 3
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

os.chdir("D:\!MOOC\Python_Directory\Data Analysis and Interpretation")

data = pd.read_csv('OOL_pds.csv', sep=',', low_memory=False)
data.columns = map(str.upper, data.columns) #upper-case all DataFrame column names
data.dtypes
data.describe()

data['OOL'] = data['W1_F6'].convert_objects(convert_numeric=True)

data['SEX'] = data['PPGENDER'].astype('category') # ft var. (binary) - gender
data['AGE'] = pd.to_numeric(data['PPAGE'], errors='coerce') # qt var. - age
data['EDU'] = pd.to_numeric(data['PPEDUCAT'], errors='coerce') # ft var. (can order) - education
data['INC'] = pd.to_numeric(data['PPINCIMP'], errors='coerce') # qt var. - household income
data['MARITAL'] = data['PPMARIT'].astype('category') #factor variable - marital status
data['ETHM'] = data['PPETHM'].astype('category')  # ft var. (can't order) - ethnic
data['SOC'] = pd.to_numeric(data['W1_P2'], errors='coerce') # ft var. (can order) - social class
data['SOC_F'] = data['W1_P3'].convert_objects(convert_numeric=True) #factor variable - social class family belong to
data['UNEMP'] = data['W1_P11'].astype('category') # ft var. (binary) - unemploy
data['OBAMA'] = data['W1_D1'].convert_objects(convert_numeric=True) #rating of [Barack Obama]
data['HILLARY'] = data['W1_D9'].convert_objects(convert_numeric=True) #rating of [Hillary Clinton]


s_data=data[['OOL','SEX','AGE','EDU','INC','MARITAL','ETHM','SOC','SOC_F','UNEMP','OBAMA','HILLARY']]
s_data.dtypes
s_data.describe() #describe quantitative var.

s_data['OOL'] = s_data['OOL'].replace(-1, np.nan)
s_data['SOC'] = s_data['SOC'].replace(-1, np.nan)
s_data['SOC_F'] = s_data['SOC_F'].replace(-1, np.nan)
s_data['UNEMP'] = s_data['UNEMP'].replace(-1, np.nan)
s_data['OBAMA'] = s_data['OBAMA'].replace(-1, np.nan)
s_data['OBAMA'] = s_data['OBAMA'].replace(998, np.nan)
s_data['HILLARY'] = s_data['HILLARY'].replace(-1, np.nan)
s_data['HILLARY'] = s_data['HILLARY'].replace(998, np.nan)

data_clean = s_data.dropna()
data_clean.dtypes
data_clean.describe()

recode1 = {1:2500, 2:6250, 3:8750, 4:11250, 5:13750, 6: 17500, 7:22500, 8: 27500, 9: 32500,
          10:37500, 11: 45000, 12:55000, 13:67500, 14: 80000, 15: 92500, 16:112500,
          17: 137500, 18: 162500, 19: 200000}
print (data_clean['INC'].value_counts(sort=False)) #before recoding
data_clean['INC']= data_clean['INC'].map(recode1)
print (data_clean['INC'].value_counts(sort=False)) #after recoding

print (data_clean['SOC'].value_counts(sort=False)) #before recoding
data_clean['SOC'] = data_clean['SOC'] -1
print (data_clean['SOC'].value_counts(sort=False)) #after recoding

recode1 = {1: 0, 2: 1}
data_clean['SEX']= data_clean['SEX'].map(recode1)
data_clean['SEX'] = data_clean['SEX'].astype('category')
print (data_clean['SEX'].value_counts(sort=False))

recode1 = {1: 1, 2: 0}
data_clean['UNEMP']= data_clean['UNEMP'].map(recode1)
data_clean['UNEMP'] = data_clean['UNEMP'].astype('category')
print (data_clean['UNEMP'].value_counts(sort=False))

data_clean.dtypes
data_clean.describe()

predvar = data_clean[['SEX','AGE','EDU','INC','MARITAL','ETHM','SOC','SOC_F','UNEMP','OBAMA','HILLARY']]
target = data_clean['OOL']
predictors=predvar.copy()
from sklearn import preprocessing
predictors['SEX']=preprocessing.scale(predictors['SEX'].astype('float64'))
predictors['AGE']=preprocessing.scale(predictors['AGE'].astype('float64'))
predictors['EDU']=preprocessing.scale(predictors['EDU'].astype('float64'))
predictors['INC']=preprocessing.scale(predictors['INC'].astype('float64'))
predictors['MARITAL']=preprocessing.scale(predictors['MARITAL'].astype('float64'))
predictors['ETHM']=preprocessing.scale(predictors['ETHM'].astype('float64'))
predictors['SOC']=preprocessing.scale(predictors['SOC'].astype('float64'))
predictors['SOC_F']=preprocessing.scale(predictors['SOC_F'].astype('float64'))
predictors['UNEMP']=preprocessing.scale(predictors['UNEMP'].astype('float64'))
predictors['OBAMA']=preprocessing.scale(predictors['OBAMA'].astype('float64'))
predictors['HILLARY']=preprocessing.scale(predictors['HILLARY'].astype('float64'))

predictors.dtypes
predictors.describe()
pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, target,
test_size=.3, random_state=123)

print (pred_train.shape)
print (pred_test.shape)
print (tar_train.shape)
print (tar_test.shape)
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

coef = dict(zip(predictors.columns, model.coef_))
import operator
sorted(coef.items(), key=operator.itemgetter(1), reverse=True)

m_log_alphas = -np.log10(model.alphas_) #alpha = penalty parameter = lambda through the model selection process
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T) #.T = transpose
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
           label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
# first 3 line = SOC, INC, EDU

m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
        label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
           label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
       
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error) #similar accuracy
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test) #more accurate than training data
