from cgi import test
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

patients = pd.read_csv("Test.csv")[:50]  # first 50 data 

# separeting target variables
x = patients.values[:,1:24]
y = patients.values[:,24:]

# splitting data set as test and train
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3,random_state=0)

# function to perform training with entropy
clfEntropy = DecisionTreeClassifier(criterion="entropy",random_state=0)
clfEntropy.fit(x_train,y_train)

# make a prediction 
y_predict = clfEntropy.predict(x_test)
print(y_predict)

# checking accuracy
print("Accuracy is ",round(accuracy_score(y_test,y_predict)*100,2),"%") 

"""
Entropy is te measurement of randomness of unpredictability int he dataset

Formula for Entropy : p(i).log2(p(i)) i = 1...k , k is the number of entropy type, p(i) is percentage of i's animal


"""