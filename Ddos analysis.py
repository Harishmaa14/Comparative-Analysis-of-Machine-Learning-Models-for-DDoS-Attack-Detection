#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("ddos.csv")


# In[3]:


X=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values


# In[4]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)


# In[6]:


import numpy as np

# Remove rows with NaN values
mask = ~np.isnan(X_train).any(axis=1)  # Create a mask for rows without NaNs
X_train = X_train[mask]
y_train = y_train[mask]  # Ensure y_train corresponds to the updated X_train|


# In[7]:


from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))


# In[8]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))


# In[9]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,y_train)
predictions = gnb.predict(X_test)
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)

predictions = knn.predict(X_test)
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))


# In[11]:


from sklearn import tree

DTClassifier = tree.DecisionTreeClassifier()
DTClassifier.fit(X_train,y_train)

print("Decision Tree Score:",DTClassifier.score(X_test,y_test))


# In[12]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
y_pred = [ 0 for i in y_pred if i < 0.5]
y_pred = [ 1 for i in y_pred if i >= 0.5]
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))


# In[13]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(alpha=1e-05,hidden_layer_sizes=(32),max_iter=1000)
mlp.fit(X_train,y_train)
y_test = mlp.predict(X_test)
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))


# In[ ]:




