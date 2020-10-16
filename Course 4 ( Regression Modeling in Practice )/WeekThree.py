import numpy
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn



data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

data['lifeoutlook'] = data['W1_F6'].convert_objects(convert_numeric=True) 
data['hh_income'] = pandas.to_numeric(data['PPINCIMP'], errors='coerce')
data['soc_class'] = pandas.to_numeric(data['W1_P2'], errors='coerce')
data['ethnicity'] = data['PPETHM'].astype('category') 


sub1=data[['lifeoutlook', 'hh_income', 'soc_class', 'ethnicity']]

sub2=sub1[(sub1['lifeoutlook'] > -1)]
print (sub1["lifeoutlook"].value_counts(sort=False))  
print (sub2["lifeoutlook"].value_counts(sort=False))  

sub3 = sub2[(sub2['soc_class'] > -1)]
print (sub2['soc_class'].value_counts(sort=False)) 
print (sub3['soc_class'].value_counts(sort=False)) 

sub4 = sub3[((sub3['ethnicity'] == 1) | (sub3['ethnicity'] == 2))]
print (sub3['ethnicity'].value_counts(sort=False)) 
print (sub4['ethnicity'].value_counts(sort=False)) 

recode1 = {1:2500, 2:6250, 3:8750, 4:11250, 5:13750, 6: 17500, 7:22500, 8: 27500, 9: 32500, 
           10:37500, 11: 45000, 12:55000, 13:67500, 14: 80000, 15: 92500, 16:112500, 
           17: 137500, 18: 162500, 19: 200000}
print (sub4["hh_income"].value_counts(sort=False)) #before recoding
sub4['hh_income']= sub4['hh_income'].map(recode1)
print (sub4["hh_income"].value_counts(sort=False))

print (sub4['soc_class'].value_counts(sort=False)) #before recoding
sub4['soc_class'] = sub4['soc_class'] - 1
print (sub4['soc_class'].value_counts(sort=False)) #after recoding

sub4['hh_income_c'] = sub4['hh_income'] - sub4['hh_income'].mean()
sub4['hh_income_c'].describe()

sub4['hh_income_c'] = sub4['hh_income_c'] / 10000
sub4['hh_income_c'].describe()

seaborn.factorplot(x="ethnicity", y="lifeoutlook", data=sub4, kind="bar", ci=None)
plt.xlabel('Ethnic Minority')
plt.ylabel('Average Life Outlook Score') #not that different

seaborn.factorplot(x="soc_class", y="lifeoutlook", data=sub4, kind="bar", ci=None)
plt.xlabel('Social Class')
plt.ylabel('Average Life Outlook Score') #significant trend*

scat1 = seaborn.regplot(x="hh_income", y="lifeoutlook", scatter=True, data=sub4)
plt.xlabel('Household Income')
plt.ylabel('Life Outlook Score') #significant trend*

scat2 = seaborn.regplot(x="hh_income", y="lifeoutlook", scatter=True, order=2, data=sub4)
plt.xlabel('Household Income')
plt.ylabel('Life Outlook Score') #significant trend*

reg1 = smf.ols('lifeoutlook ~ hh_income_c', data=sub4).fit()
print (reg1.summary())

reg2 = smf.ols('lifeoutlook ~ hh_income_c + I(hh_income_c**2)', data=sub4).fit()
print (reg2.summary()) #Adj. R-squared change little (0.110 -> 0.116) -> quadratic insignificant

reg3 = smf.ols('lifeoutlook ~ hh_income_c + soc_class', data=sub4).fit()
print (reg3.summary()) #Adj. R-squared change significantly (0.110 -> 0.204)

reg4 = smf.ols('lifeoutlook ~ hh_income_c + soc_class + ethnicity', data=sub4).fit()
print (reg4.summary()) #Adj. R-squared change little (0.204 -> 0.206) -> insignificant


fig1 = sm.qqplot(reg3.resid, line='r') 

stdres=pandas.DataFrame(reg3.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=2, color='r')
l = plt.axhline(y=0, color='r')
l = plt.axhline(y=-2, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
fig2 = plt.figure(figsize=(8,6))
fig2 = sm.graphics.plot_regress_exog(reg3, "hh_income_c", fig=fig2)
print(fig2)

fig3 = sm.graphics.influence_plot(reg3, size=3)
l = plt.axhline(y=2, color='r')
l = plt.axhline(y=-2, color='r')
print(fig3)
