import library
import pandas
import numpy

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import seaborn
import matplotlib.pyplot as plt


data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

data['PPETHM'] = data['PPETHM'].astype('category') #factor variable - ethnic
data['W1_D1'] = data['W1_D1'].convert_objects(convert_numeric=True) #factor variable
data['W1_D9'] = data['W1_D9'].convert_objects(convert_numeric=True) #factor variable


data['W1_F6'] = data['W1_F6'].convert_objects(convert_numeric=True) #factor variable - life outlook (quantitative)


sub1=data[['PPETHM', 'W1_D1','W1_D9', 'W1_F6', 'PPAGECT4']]
sub1=sub1[(data['PPAGECT4']==1) | (data['PPAGECT4']==2)]
sub2=sub1.copy()
sub2['W1_D1']=sub2['W1_D1'].replace(-1, numpy.nan)
sub2['W1_D1']=sub2['W1_D1'].replace(998, numpy.nan)
sub2['W1_D9']=sub2['W1_D9'].replace(-1, numpy.nan)
sub2['W1_D9']=sub2['W1_D9'].replace(998, numpy.nan)
sub2['W1_F6']=sub2['W1_F6'].replace(-1, numpy.nan)



print (sub2['W1_D1'].value_counts(sort=False, dropna=False).sort_index())
print (sub2['W1_D9'].value_counts(sort=False, dropna=False).sort_index())
print (sub2['W1_F6'].value_counts(sort=False, dropna=False).sort_index())



recode1 = {1: '1_White', 2: '2_Black', 3: '4_Other', 4: '3_Hispanic', 5: '5_2+Ethnic'}
sub2['PPETHM']= sub1['PPETHM'].map(recode1)
ct1 = sub2.groupby('PPETHM').size()
print (ct1)



print ('-------OLS Results for Hilary Score Using ethinicty as factor variable-------')
model1 = smf.ols(formula='W1_D9 ~ C(PPETHM)', data=sub2)
results1 = model1.fit()
print (results1.summary())
sub3 = sub2[['W1_D9', 'PPETHM']].dropna()
print ('means for W1_D9 by ethnicity')
m1= sub3.groupby('PPETHM').mean()
print (m1)
print ('standard deviations for W1_D9 by ethnicity')
sd1 = sub3.groupby('PPETHM').std()
print (sd1)
mc1 = multi.MultiComparison(sub3['W1_D9'], sub3['PPETHM'])
res1 = mc1.tukeyhsd()

print ('------Ad-hoc Test Sesults for Comparing Score Among Ethnicity---------')
print(res1.summary())

seaborn.factorplot(x='PPETHM', y='W1_D9', data=m1, kind="bar", ci=None)
plt.xlabel('Hilary Rating')
plt.ylabel('Ethinicity')
