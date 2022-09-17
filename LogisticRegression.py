#Import libraries and their associated methods
from calendar import c
import logging
from string import digits
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
digits = load_digits() 

# determing the total number of images and labels
print("Image Data Shape ",digits.data.shape)
print("Label Data Shape ",digits.target.shape)

# displaying some of the images and labels
plt.figure(figsize=(20,4))
for index, (image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)), cmap=plt.cm.gray)
    plt.title('Trainning: %i \n' %label, fontsize = 20)
plt.show()    

# dividing dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.2,random_state=2)

# making an instance of the model and training it
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train,y_train)

# predicting the output of the first element of the test set
# print(logisticRegr.predict(x_test[0].reshape(1,-1)))

# predicting the  output of the first 10 elements of the test set
# print(logisticRegr.predict(x_test[:10]))

# predicting for the entire dataset
predictions = logisticRegr.predict(x_test)
print(predictions) 

# determining the accuracy of the model 
accuracy = logisticRegr.score(x_test,y_test)
print(round(accuracy,3)) 



# the confusion matrix 
cm = metrics.confusion_matrix(y_test,predictions)
print(cm)


"""   Logistic Resgession 
  It is a classification algorithm, used tı predict binary outcomes for a given set og inedpendent variables.The dependent variable's 
  outcome is discreate.

   The Math Behind Logistic Regression 
   To understand it, we talk about the odds of success 

       adds(thetra) = (probability of an event happening)/(probability of an event not happening)   or  thetra = p/(1-p) 

   The values of odds range from 0 to infinity
   The values of probability change from 0 to 1  

  """