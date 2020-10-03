import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt


data = pandas.read_csv('ool_pds.csv', sep=',', low_memory=False)

data2 = pandas.read_csv('gapminder.csv', low_memory=False)


data['PPAGECT4'] = data['PPAGECT4'].astype('category') #factor variable - age category
data['PPETHM'] = data['PPETHM'].astype('category') #factor variable - ethnic
data['W1_D1'] = data['W1_D1'].convert_objects(convert_numeric=True) #numeric variable
data['W1_D9'] = data['W1_D9'].convert_objects(convert_numeric=True) #numeric variable
data['W1_P2'] = data['W1_P2'].convert_objects(convert_numeric=True) #factor variable - social class belong to
data['W1_P3'] = data['W1_P3'].convert_objects(convert_numeric=True) #factor variable - social class family belong to
data['W1_F1'] = data['W1_F1'].astype('category') #factor variable

['W1_F6'] = data['W1_F6'].convert_objects(convert_numeric=True) #factor variable - life outlook (quantitative)

data2['internetuserate'] = data2['internetuserate'].convert_objects(convert_numeric=True)
data2['urbanrate'] = data2['urbanrate'].convert_objects(convert_numeric=True)
data2['incomeperperson'] = data2['incomeperperson'].convert_objects(convert_numeric=True)
data2['hivrate'] = data2['hivrate'].convert_objects(convert_numeric=True


sub1=data[['PPAGECT4','PPETHM', 'W1_D1', 'W1_D9','W1_P2', 'W1_P3', 'W1_F1', 'W1_F6']]


sub1=sub1[(data['PPAGECT4']==1) | (data['PPAGECT4']==2)]

sub2=sub1.copy()


sub2['W1_P2']=sub2['W1_P2'].replace(-1, numpy.nan)
sub2['W1_P3']=sub2['W1_P3'].replace(-1, numpy.nan)
sub2['W1_F1']=sub2['W1_F1'].replace(-1, numpy.nan)
sub2['W1_D1']=sub2['W1_D1'].replace(-1, numpy.nan)
sub2['W1_D1']=sub2['W1_D1'].replace(998, numpy.nan)
sub2['W1_D9']=sub2['W1_D9'].replace(-1, numpy.nan)
sub2['W1_D9']=sub2['W1_D9'].replace(998, numpy.nan)
sub2['W1_F6']=sub2['W1_F6'].replace(-1, numpy.nan)
data2['incomeperperson']=data2['incomeperperson'].replace(' ', numpy.nan)

recode1 = {1: 1, 2: 0, 3: -1}
sub2['W1_F1n']= sub2['W1_F1'].map(recode1)
type (sub2['W1_F1n'])


recode1 = {1: '1_White', 2: '2_Black', 3: '4_Other', 4: '3_Hispanic', 5: '5_2+Ethnic'}
sub2['PPETHM']= sub1['PPETHM'].map(recode1)



def POSITIVE (row):

  if row['W1_F1n'] == 1 :

     return 1

  elif (row['W1_F1n'] == 0) | (row['W1_F1n'] == -1) :

     return 0

  else :

      return numpy.nan

sub2['POSITIVE'] = sub2.apply (lambda row: POSITIVE (row),axis=1)


def NEGATIVE (row):

  if row['W1_F1n'] == -1 :

     return 1

  elif (row['W1_F1n'] == 0) | (row['W1_F1n'] == 1) :

     return 0    

  else :

      return numpy.nan    

sub2['NEGATIVE'] = sub2.apply (lambda row: NEGATIVE (row),axis=1)


print ("PPAGECT4")

print (sub2["PPAGECT4"].value_counts(sort=False))

print ("PPETHM")

print (sub2["PPETHM"].value_counts(sort=False))

print ("W1_P2")

print (sub2["W1_P2"].value_counts(sort=False))

print ("W1_P3")

print (sub2["W1_P3"].value_counts(sort=False))

print ("W1_F1n")

print (sub2["W1_F1n"].value_counts(sort=False))

print ("POSITIVE")

print (sub2["POSITIVE"].value_counts(sort=False))

print ("NEGATIVE")

print (sub2["NEGATIVE"].value_counts(sort=False))

#3. Practive Plotting Data

#3.1 Univariate bar graph for categorical variables

sub2["PPETHM"] = sub2["PPETHM"].astype('category')

seaborn.countplot(x="PPETHM", data=sub2)

plt.xlabel('Ethnic Minority')

plt.title('Ethnic Minority among person age 18-44 years in the OOL Study')

#3.2 Univariate histogram for quantitative variable:

seaborn.distplot(sub2["W1_D1"].dropna(), kde=False);

plt.xlabel('Rating Score for Barack Obama')

plt.title('Rating Score for Barack Obama among person age 18-44 years in the OOL Study')


seaborn.factorplot(x="PPETHM", y="W1_D1", data=sub2, kind="bar", ci=None)

plt.xlabel('Ethnic Minority')

plt.ylabel('Average Rating Score for Barck Obama')


sub3 = sub2[['W1_D1', 'PPETHM']].dropna()

print ('means for Rating Score for Barck Obama by ethnicity')

m1= sub3.groupby('PPETHM').mean()

print (m1) #correct


seaborn.factorplot(x='PPETHM', y='NEGATIVE', data=sub2, kind="bar", ci=None)

plt.xlabel('Ethnic Minority')

plt.ylabel('Proportion of Negative Outlook')


sub4 = sub2[['NEGATIVE', 'PPETHM']].dropna()

print ('means for NEGATIVE by ethnicity')

m2 = sub4.groupby('PPETHM').mean()

print (m2) #correct


scat1 = seaborn.regplot(x="W1_D1", y="W1_D9", data=sub2)

plt.xlabel('Rating Score for Barack Obama')

plt.ylabel('Rating Score for Hillary Clinton')

plt.title('Scatterplot for the Association Between Obama and Hillary Rating Score') #somewhat positive relationship


scat2 = seaborn.regplot(x="W1_D1", y="W1_F6", data=sub2)

plt.xlabel('Rating Score for Barack Obama')

plt.ylabel('Rating Score for Life Outllok')

plt.title('Scatterplot for the Association Between President Likeness and Life Outlook') #no clear relationship


scat3 = seaborn.regplot(x="urbanrate", y="internetuserate", fit_reg=False, data=data2)

plt.xlabel('Urban Rate')

plt.ylabel('Internet Use Rate')

plt.title('Scatterplot for the Association Between Urban Rate and Internet Use Rate') #positive relationship
