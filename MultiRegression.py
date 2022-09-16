"""In this process, a established between relationship is independent and dependent 
   variables by fitting them to a line.This line is known as the regression line and
   represented by a linear equation y = a*x + b  """
import pandas as pd
from sklearn import linear_model

data = pd.read_csv("Test.csv")
 
# all data will be numerical value
d = {'Low':0,'Medium':1,'High':2}
data['Level'] = data['Level'].map(d)

# x --> independent variables  y --> dependent variables 
x = data[['Air Pollution','Alcohol use','chronic Lung Disease','Obesity','Smoking']]
y = data['Level']

# From the sklearn module it is used the LinearRegression() method to create a linear regression obj.0
linReg_obj = linear_model.LinearRegression()  
linReg_obj.fit(x,y) 

predicTheLevel = linReg_obj.predict([[2,4,2,4,3]]) 

print(predicTheLevel)
