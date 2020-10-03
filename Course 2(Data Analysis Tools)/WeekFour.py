import numpy
import pandas
import scipy.stats
import statsmodels.formula.api as smf
import seaborn
import matplotlib.pyplot as plt

data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

data['W1_D1'] = data['W1_D1'].convert_objects(convert_numeric=True)
data['W1_P2'] = data['W1_P2'].convert_objects(convert_numeric=True)
data['PPAGECT4'] = data['PPAGECT4'].astype('category') #factor variable - age category

sub1=data[['W1_D1','W1_P2','PPAGECT4']]

sub1['W1_D1']=sub1['W1_D1'].replace(-1, numpy.nan)
sub1['W1_D1']=sub1['W1_D1'].replace(998, numpy.nan)
sub1['W1_P2']=sub1['W1_P2'].replace(-1, numpy.nan)
recode3 = {1: '1_Young+Middle', 2: '1_Young+Middle', 3: '2_Elder', 4: '2_Elder'}
sub1['AGROUP']= sub1['PPAGECT4'].map(recode3)
sub1['AGROUP']=sub1['AGROUP'].astype('category')
sub2=sub1[(sub1['AGROUP']=='1_Young+Middle')]
sub3=sub1[(sub1['AGROUP']=='2_Elder')]

sub21 = sub2[['W1_D1', 'W1_P2']].dropna()
sub31 = sub3[['W1_D1', 'W1_P2']].dropna()

print ('Association between social class and Obama score for YOUNG&MIDDLE-AGE person')
print (scipy.stats.pearsonr(sub21['W1_D1'], sub21['W1_P2'])) #signicant negative 

print ('Association between social class and Obama score for ELDER person')
print (scipy.stats.pearsonr(sub31['W1_D1'], sub31['W1_P2'])) #insignificant

scat1 = seaborn.regplot(x="W1_P2", y="W1_D1", data=sub21)
plt.xlabel('Social class')
plt.ylabel('Obama score')
plt.title('Scatterplot for the association between social class and Obama score for YOUNG&MIDDLE-AGE person')
print (scat1)

scat2 = seaborn.regplot(x="W1_P2", y="W1_D1", data=sub31)
plt.xlabel('Social class')
plt.ylabel('Obama score')
plt.title('Scatterplot for the association between social class and Obama score for ELDER person')
print (scat2)

sub23 = sub2[['W1_D1', 'W1_D9']].dropna()
sub33 = sub3[['W1_D1', 'W1_D9']].dropna()
sub43 = sub4[['W1_D1', 'W1_D9']].dropna()

print ('Association between Obama and Hillary rating score for LOWER social class')
print (scipy.stats.pearsonr(sub23['W1_D1'], sub23['W1_D9']))

print ('Association between Obama and Hillary rating score for MIDDLE social class')
print (scipy.stats.pearsonr(sub33['W1_D1'], sub33['W1_D9']))

print ('Association between Obama and Hillary rating score for UPPER social class')
print (scipy.stats.pearsonr(sub43['W1_D1'], sub43['W1_D9']))

scat1 = seaborn.regplot(x="W1_D9", y="W1_D1", data=sub23)
plt.xlabel('Hillary score')
plt.ylabel('Obama score')
plt.title('Scatterplot for the association between Obama and Hillary rating score for LOWER social class')
print (scat1)

scat2 = seaborn.regplot(x="W1_D9", y="W1_D1", data=sub33)
plt.xlabel('Hillary score')
plt.ylabel('Obama score')
plt.title('Scatterplot for the association between Obama and Hillary rating score for MIDDLE social class')
print (scat2)

scat3 = seaborn.regplot(x="W1_D9", y="W1_D1", data=sub43)
plt.xlabel('Hillary score')
plt.ylabel('Obama score')
plt.title('Scatterplot for the association between Obama and Hillary rating score for UPPER social class')
print (scat3)

sub23 = sub2[['W1_D1', 'W1_F6']].dropna()
sub33 = sub3[['W1_D1', 'W1_F6']].dropna()
sub43 = sub4[['W1_D1', 'W1_F6']].dropna()

print ('Association between Obama and Hillary rating score for LOWER social class')
print (scipy.stats.pearsonr(sub23['W1_D1'], sub23['W1_D9']))

print ('Association between Obama and Hillary rating score for MIDDLE social class')
print (scipy.stats.pearsonr(sub33['W1_D1'], sub33['W1_D9']))

print ('Association between Obama and Hillary rating score for UPPER social class')
print (scipy.stats.pearsonr(sub43['W1_D1'], sub43['W1_D9']))

scat1 = seaborn.regplot(x="W1_F6", y="W1_D1", data=sub23)
plt.xlabel('Hillary score')
plt.ylabel('Life Outlook')
plt.title('Scatterplot for the association between Obama score and life outlook for LOWER social class')
print (scat1)

scat2 = seaborn.regplot(x="W1_F6", y="W1_D1", data=sub33)
plt.xlabel('Hillary score')
plt.ylabel('Life Outlook')
plt.title('Scatterplot for the association between Obama score and life outlook for MIDDLE social class')
print (scat2)

scat3 = seaborn.regplot(x="W1_F6", y="W1_D1", data=sub43)
plt.xlabel('Hillary score')
plt.ylabel('Life Outlook')
plt.title('Scatterplot for the association between Obama score and life outlook for UPPER social class')
print (scat3)

recode1 = {1: 1, 2: 0, 3: -1}
sub1['W1_F1n']= sub1['W1_F1'].map(recode1)
type (sub1['W1_F1n'])

recode2 = {1: '1_White', 2: '2_Black', 3: '4_Other', 4: '3_Hispanic', 5: '5_2+Ethnic'}
sub1['PPETHM']= sub1['PPETHM'].map(recode2)

recode3 = {1: '1_Lower', 2: '1_Lower', 3: '2_Middle', 4: '3_Upper', 5: '3_Upper'}
sub1['SOCIAL']= sub1['W1_P2'].map(recode3)
sub1['SOCIAL']=sub1['SOCIAL'].astype('category')
data['PPETHM'] = data['PPETHM'].astype('category') #factor variable - ethnic

data['W1_F1'] = data['W1_F1'].astype('category') #factor variable
data['W1_D9'] = data['W1_D9'].convert_objects(convert_numeric=True) #numeric variable
data['W1_F6'] = data['W1_F6'].convert_objects(convert_numeric=True) #factor variable - life outlook (quantitative)

def NEGATIVE (row):
  if row['W1_F1n'] == -1 :
     return 1
  elif (row['W1_F1n'] == 0) | (row['W1_F1n'] == 1) :
     return 0    
  else :
      return numpy.nan    
sub1['NEGATIVE'] = sub1.apply (lambda row: NEGATIVE (row),axis=1)

sub2=sub1[(sub1['SOCIAL']=='1_Lower')]
sub3=sub1[(sub1['SOCIAL']=='2_Middle')]
sub4=sub1[(sub1['SOCIAL']=='3_Upper')]


sub21 = sub2[['W1_D1', 'PPETHM']].dropna()
sub31 = sub3[['W1_D1', 'PPETHM']].dropna()
sub41 = sub4[['W1_D1', 'PPETHM']].dropna()

print ("Means for Obama rating score by Ethnicity for lower social class")
m1= sub21.groupby('PPETHM').mean()
print (m1)

print ("Means for Obama rating score by Ethnicity for middle social class")
m2= sub31.groupby('PPETHM').mean()
print (m2)

print ("Means for Obama rating score by Ethnicity for upper social class")
m3= sub41.groupby('PPETHM').mean()
print (m3)

# 3.1.3 ANOVA Analysis
print ('Association between ethnicity and Obama rating score for those in LOWER social class')
model1 = smf.ols(formula='W1_D1 ~ C(PPETHM)', data=sub21).fit()
print (model1.summary())

print ('Association between ethnicity and Obama rating score for those in MIDDLE social class')
model2 = smf.ols(formula='W1_D1 ~ C(PPETHM)', data=sub31).fit()
print (model2.summary())

print ('Association between ethnicity and Obama rating score for those in UPPER social class')
model3 = smf.ols(formula='W1_D1 ~ C(PPETHM)', data=sub41).fit()
print (model3.summary())

seaborn.factorplot(x="PPETHM", y="W1_D1", data=sub21, kind="bar", ci=None,
                  order=["2_Black", "3_Hispanic", "4_Other", "1_White", "5_2+Ethnic"])
plt.xlabel('Ethnicity')
plt.ylabel('Obama rating score')

seaborn.factorplot(x="PPETHM", y="W1_D1", data=sub31, kind="bar", ci=None,
                  order=["2_Black", "4_Other", "5_2+Ethnic", "3_Hispanic", "1_White"])
plt.xlabel('Ethnicity')
plt.ylabel('Obama rating score')

seaborn.factorplot(x="PPETHM", y="W1_D1", data=sub41, kind="bar", ci=None,
                  order=["2_Black", "3_Hispanic", "4_Other", "1_White", "5_2+Ethnic"])
plt.xlabel('Ethnicity')
plt.ylabel('Obama rating score')

