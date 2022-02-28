#!/usr/bin/env python
# coding: utf-8

# In[240]:


import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


df, meta = pyreadstat.read_sav(r'C:\Users\90536\Downloads\saadHS1.sav')
df2, meta = pyreadstat.read_sav(r'C:\Users\90536\Downloads\saadUS1.sav')


# In[241]:


df


# In[ ]:





# In[242]:


df2


# In[243]:


null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()


# In[244]:


data=df.apply (pd.to_numeric, errors='coerce')
data=df.fillna(0)
df.update(data)
df


# In[ ]:





# In[245]:


null_columns=df2.columns[df2.isnull().any()]
df2[null_columns].isnull().sum()


# In[246]:


data=df2.apply (pd.to_numeric, errors='coerce')
data=df2.fillna(0)

df2.update(data)
df2


# In[247]:


null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()


# In[ ]:





# In[ ]:





# In[248]:


df2.head()


# In[249]:


null_columns=df2.columns[df2.isnull().any()]
df2[null_columns].isnull().sum()


# In[250]:


categorical_features=[feature for feature in df.columns if df[feature].dtype=='O']
categorical_features


# In[251]:


categorical_features2=df.select_dtypes(include=np.number).columns.tolist()
categorical_features2


# In[252]:


categorical_features=[feature for feature in df2.columns if df2[feature].dtype=='O']
categorical_features


# In[253]:


categorical_features2=df2.select_dtypes(include=np.number).columns.tolist()
categorical_features2


# In[254]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = df2.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df2[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[255]:


X = df2.drop(['CGPA','UniversityDepartment'],1) 
X


# In[256]:


Y = df2['UniversityDepartment']


# In[257]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn import tree


# In[258]:


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[259]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=10)
clf_gini.fit(X_train, y_train)


# In[260]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=10)
clf_entropy.fit(X_train, y_train)


# In[261]:


#clf_gini.predict([3, 3, 1, 1])
X_test.shape


# In[262]:


y_pred = clf_gini.predict(X_test)
y_pred


# In[263]:



print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)


# In[264]:


clf = tree.DecisionTreeClassifier(random_state=100)
clf = clf.fit(X_train, y_train)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf)
tree.plot_tree(clf_gini, precision=0)


# In[265]:


y_pred = clf.predict(X_test)
y_pred
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)


# In[266]:



y_pred = clf_entropy.predict(X_test)
y_pred
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)


# In[267]:


clf


# In[268]:


clf_gini


# In[269]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
 
predictions = logmodel.predict(X_test)
predictions


# In[270]:


predictions = logmodel.predict(X_test)
#print(classification_report(y_test, predictions))
#print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))


# In[271]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
 
#predictions = logmodel.predict(df)
#print(accuracy_score(y_test, predictions))


new_output = logmodel.predict(df)
# summarize input and output
# for i in range(len(new_output)):
#     print(new_output[i])
#  print( new_output[1])
d=df.to_numpy()
temp =[]
for i in range(len(df)):
    temp.append(new_output[i])
#     print(d[i],"$$$$$$",new_output[i])
# temp
df["final"]=temp
df.head(92)


# In[272]:


x = df['final']==6
x.sum()


# In[273]:


X_test.head(2)


# In[274]:


predictions[29]


# In[275]:


params = logmodel.get_params()
print(params)


# In[276]:


print('Intercept: \n', logmodel.intercept_)


# In[277]:


sns.heatmap(pd.DataFrame(confusion_matrix(y_test,predictions)))
plt.show()


# In[278]:


# RMSE


# In[279]:


from sklearn.metrics import mean_squared_error
from math import sqrt


# In[280]:


y_predicted = logmodel.predict(X)
y_predicted


# In[281]:


rms = sqrt(mean_squared_error(Y, y_predicted))
rms


# In[282]:


rmse = sqrt(mean_squared_error(Y, y_predicted,squared=False))
rmse


# In[283]:


# MAE


# In[284]:


from sklearn.metrics import mean_absolute_error


# In[285]:


mae = mean_absolute_error(Y,y_predicted)
mae


# In[286]:


y_predicted =clf.predict(X)
y_predicted


# In[287]:


rms = sqrt(mean_squared_error(Y, y_predicted))
rms


# In[288]:


rmse = sqrt(mean_squared_error(Y, y_predicted,squared=False))
rmse


# In[289]:


from sklearn.metrics import mean_absolute_error


# In[290]:


mae = mean_absolute_error(Y,y_predicted)
mae


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




