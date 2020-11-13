import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
mnist_data = pd.read_csv("data.csv")
data = mnist_data.copy()
X, y = data.drop(labels = ["label"],axis = 1).to_numpy(), data["label"]
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
y_train_8 = (y_train == 8)
y_test_8 = (y_test == 8)
sgd_clf = SGDClassifier(max_iter=1000,random_state = 42)
sgd_clf.fit(X_train, y_train_8)
y_test_pred=sgd_clf.predict(X_test)
print(y_test_pred)
print(y_test_8)
 

Confusion=confusion_matrix(y_test_8, y_test_pred)
print(Confusion)

Accuracy=(Confusion[0,0]+Confusion[1,1])/(Confusion[1,1]+Confusion[1,0]+Confusion[0,1] +Confusion[0,0]) #Number of correct classified points / Total number of points 
print(Accuracy) #0.9433333333333334

multiclass=sgd_clf.fit(X_train,y_train)
y_test_predict=sgd_clf.predict(X_test)
print(y_test_predict)
