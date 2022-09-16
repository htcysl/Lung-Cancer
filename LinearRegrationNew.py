""" Simple Linear Regression  =>  y = m*x+n 
    Multiple Linear Regression =>  y =m1*x1 + m2*x2+ .... + mf*xf  + n  """

from cgi import test
from unicodedata import category
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

patients = pd.read_csv("Test.csv")

X = patients.iloc[:,1:-1].values
y = patients.iloc[:,24].values

# data visialization
# sns.heatmap(patients.corr) ?!

# Encoding categorical data 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# splitting the data into  train and test data sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0) 

# **fitting multiple linear regression model to traing set**
from sklearn.linear_model import LinearRegression
regr = LinearRegression()   # regression obj.
regr.fit(X_train,y_train)

# predicting the test set result
y_predict = regr.predict(X_test)  

# calculating the coefficient 
print(regr.coef_)

# calculating the intercept
print(regr.intercept_)

# Evaluating the model; calc. the R squared value ; 
from sklearn.metrics import r2_score
print(r2_score(y_test,y_predict)) 


