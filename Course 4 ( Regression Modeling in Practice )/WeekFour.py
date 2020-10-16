import numpy
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn

data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

data['outlook_qt'] = data['W1_F6'].convert_objects(convert_numeric=True) 
data['outlook_ql'] = data['W1_F1'].astype('category') #factor variable 

data['hh_income'] = pandas.to_numeric(data['PPINCIMP'], errors='coerce')
data['soc_class'] = pandas.to_numeric(data['W1_P2'], errors='coerce')
data['ethnicity'] = data['PPETHM'].astype('category') 


sub1=data[['outlook_ql', 'hh_income', 'soc_class', 'ethnicity']]

sub1['outlook_ql']=sub1['outlook_ql'].replace(-1, numpy.nan) 
sub1['soc_class']=sub1['soc_class'].replace(-1, numpy.nan) 

recode1 = {1: 1, 2: 0, 3: -1}
sub1['outlook_ql']= sub1['outlook_ql'].map(recode1)
type (sub1['outlook_ql'])

recode1 = {1:2500, 2:6250, 3:8750, 4:11250, 5:13750, 6: 17500, 7:22500, 8: 27500, 9: 32500, 
           10:37500, 11: 45000, 12:55000, 13:67500, 14: 80000, 15: 92500, 16:112500, 
           17: 137500, 18: 162500, 19: 200000}
print (sub1["hh_income"].value_counts(sort=False)) #before recoding
sub1['hh_income']= sub1['hh_income'].map(recode1)
print (sub1["hh_income"].value_counts(sort=False)) #after recoding

print (sub1['soc_class'].value_counts(sort=False)) #before recoding
sub1['soc_class'] = sub1['soc_class'] - 1
print (sub1['soc_class'].value_counts(sort=False)) #after recoding

def NEGATIVE (row):
   if row['outlook_ql'] == -1 : 
      return 1 
   elif (row['outlook_ql'] == 0) | (row['outlook_ql'] == 1) :
      return 0    
   else :
       return numpy.nan     
sub1['neg_outlook'] = sub1.apply (lambda row: NEGATIVE (row),axis=1)

print (sub1["outlook_ql"].value_counts(sort=False)) 
print (sub1["neg_outlook"].value_counts(sort=False)) 

sub1['hh_income_c'] = sub1['hh_income'] - sub1['hh_income'].mean()
sub1['hh_income_c'].describe()

sub1['hh_income_c'] = sub1['hh_income_c'] / 10000
sub1['hh_income_c'].describe()

reg1 = smf.ols('neg_outlook ~ hh_income_c', data=sub1).fit()
print (reg1.summary()) #Adj. R-squared = -0.000 #household income insignificant

reg2 = smf.ols('neg_outlook ~ soc_class', data=sub1).fit()
print (reg2.summary()) #Adj. R-squared = 0.002

reg3 = smf.ols('neg_outlook ~ soc_class + C(ethnicity)', data=sub1).fit()
print (reg3.summary()) #Adj. R-squared = 0.002 -> 0.045



lreg1 = smf.logit(formula = 'neg_outlook ~ hh_income_c', data = sub1).fit()
print (lreg1.summary()) #household income insignificant
print ("Odds Ratios")
print (numpy.exp(lreg1.params))

lreg2 = smf.logit(formula = 'neg_outlook ~ soc_class', data = sub1).fit()
print (lreg2.summary())

params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))

lreg3 = smf.logit(formula = 'neg_outlook ~ soc_class + C(ethnicity)', data = sub1).fit()
print (lreg3.summary())

print ("Odds Ratios")
params = lreg3.params
conf = lreg3.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))
