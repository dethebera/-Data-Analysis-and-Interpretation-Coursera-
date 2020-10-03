import pandas
import numpy
import seaborn
import scipy
import matplotlib.pyplot as plt

data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

data['W1_D1'] = data['W1_D1'].convert_objects(convert_numeric=True)
data['W1_D9'] = data['W1_D9'].convert_objects(convert_numeric=True)
data['W1_P2'] = data['W1_P2'].convert_objects(convert_numeric=True)
data['W1_F6'] = data['W1_F6'].convert_objects(convert_numeric=True)

sub1=data[['PPAGECT4','W1_D1','W1_D9','W1_P2','W1_F6']]
sub1=sub1[(data['PPAGECT4']==1) | (data['PPAGECT4']==2)]
sub2=sub1.copy()

sub2['W1_D1']=sub2['W1_D1'].replace(-1, numpy.nan)
sub2['W1_D1']=sub2['W1_D1'].replace(998, numpy.nan)
sub2['W1_D9']=sub2['W1_D9'].replace(-1, numpy.nan)
sub2['W1_D9']=sub2['W1_D9'].replace(998, numpy.nan)
sub2['W1_P2']=sub2['W1_P2'].replace(-1, numpy.nan)
sub2['W1_F6']=sub2['W1_F6'].replace(-1, numpy.nan)

print ("W1_D1: Obama Score")
print (sub2["W1_D1"].value_counts(sort=False))

print ("W1_D9: Hillary Score")
print (sub2["W1_D9"].value_counts(sort=False))

print ("W1_P2: Social Class")
print (sub2["W1_P2"].value_counts(sort=False))

print ("W1_F6: Life Outlook")
print (sub2["W1_F6"].value_counts(sort=False))

sub_clean = sub2.dropna() #drop na to calculate correlation

scat1 = seaborn.regplot(x="W1_D1", y="W1_D9", data=sub_clean)
plt.xlabel('Rating Score for Barack Obama')
plt.ylabel('Rating Score for Hillary Clinton')
plt.title('Scatterplot between Obama and Hillary Rating Score') #somewhat positive relationship

print ('association between Obama and Hillary Rating Score')
print (scipy.stats.pearsonr(sub_clean['W1_D1'], sub_clean['W1_D9']))
scat2 = seaborn.regplot(x="W1_D1", y="W1_F6", data=sub_clean)

plt.xlabel('Rating Score for Barack Obama')
plt.ylabel('Rating Score for Life Outllok')
plt.title('Scatterplot between Obama Score and Life Outlook') #no clear relationship

print ('association between Obama Score and Life Outlook')
print (scipy.stats.pearsonr(sub_clean['W1_D1'], sub_clean['W1_F6']))

scat3 = seaborn.regplot(x="W1_P2", y="W1_D1", data=sub_clean)
plt.xlabel('Social Class')
plt.ylabel('Rating Score for Barack Obama')
plt.title('Scatterplot between Social Class and Obama Score') #somewhat negative relationship

print ('association between Social Class and Obama Scor')
print (scipy.stats.pearsonr(sub_clean['W1_P2'], sub_clean['W1_D1']))
