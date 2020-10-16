
import pandas
import statsmodels.formula.api as smf
import seaborn
import matplotlib.pyplot as plt

data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

data['W1_P2'] = pandas.to_numeric(data['W1_P2'], errors='coerce') 
data['W1_F6'] = pandas.to_numeric(data['W1_F6'], errors='coerce') 

sub1=data[['W1_P2', 'W1_F6']]

sub2=sub1[(sub1['W1_P2'] > -1)]
print (sub2["W1_P2"].value_counts(sort=False))

recode1 = {1: 0, 2: 0, 3:0, 4:1, 5:1}
sub2['W1_P2']= sub2['W1_P2'].map(recode1)
print (sub2["W1_P2"].value_counts(sort=False))

plot1 = seaborn.factorplot(x="W1_P2", y="W1_F6", data=sub2, kind="bar", ci=None)
plt.xlabel('Social Class (0: Middle and Below vs 1: Higher than Middle)')
plt.ylabel('Mean of Life Outlook Score')
plt.title ('Bivariate Bar Graph for the Social Class and Life Outlook Score')
print(plot1)

print ("OLS Regression Model for the Association Between Social Class and Life Outlook Score")
reg1 = smf.ols('W1_F6 ~ W1_P2', data=sub2).fit()
print (reg1.summary())
