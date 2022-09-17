import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

bank = pd.read_csv("Test.csv")[:50]  # first 50 data 

print("Data set shape :",bank.shape)

