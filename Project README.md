# Iris-Project

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:29:10 2022

@author: ASUS
"""

import pandas as pd
import numpy as np

a=pd.read_csv(r'C:\Users\ASUS\Desktop\AI_ML\iris.csv')
a.shape
a.size
a.ndim
a.info()
a.describe()

a['pw'].fillna(a['pw'].mean(),inplace=True)
"""
x=a.iloc[:,:-1]
y=a.iloc[:,-1]

x.shape
type(x)

y.shape
type(y)
"""

x=a.iloc[:,:-1].values

x

y=a.iloc[:,-1].values

y

type(x)

type(y)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

print(model.predict([[5.2,4.1,1.6,1.4]]))

print(model.predict([[6.5,5.2,3.7,2.5]]))

print(model.predict([[6.3,5.8,1.7,1.1]]))

print(model.predict([[10,8,5,4.0]]))
