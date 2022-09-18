# Loading the library with the iris dataset
from sklearn.datasets import load_iris

# Loading scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
# Setting random seed
np.random.seed(0)

# Creating an object called iris with the iris data
iris = load_iris()

# Creating a dataframe with the four feature variables
df = pd.DataFrame(iris.data,columns=iris.feature_names)

# Adding a new column for the species name
df['species'] = pd.Categorical.from_codes(iris.target,iris.target_names)
df['is_train'] = np.random.uniform(0,1,len(df)) <= .75

# Creating dataframe with rows and training rows
train, test = df[df['is_train'] == True],df[df['is_train']==False] 

# Show the number of observations for the test and training dataframe
#print("Number of observations in the training data :",len(train))
#print("Number of observations in the test data : ",len(test))

# create a list of the feature column's names
features = df.columns[:4] 

# Coverting each species name into digits
y = pd.factorize(train['species'])[0]

# Creating a random forest classifier 
objClf = RandomForestClassifier(n_jobs=2,random_state=0)

# train the classifier
objClf.fit(train[features],y)

# Applying the trained classifier to the test
predict = objClf.predict(test[features])

#Viewing the predicted probabilities of the first 10 observations 
#print(objClf.predict_proba(test[features])[0:10])

# mapping names for the plants for each predicted plant class 
preds = iris.target_names[objClf.predict(test[features])]

#view the predicted sspecies for the first five observations 
#print(preds[:5])

# Creating confusion matrix
print(pd.crosstab(test['species'],preds,rownames=['Actual Species'],colnames=['Predicted Species']) )