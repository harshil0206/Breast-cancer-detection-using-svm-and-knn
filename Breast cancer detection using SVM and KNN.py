import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10000)

url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names=["id", "clump_thickness", "uniform_cell_size", "uniform_cell_shape", "marginal_adhesion",
       "signle_epithelial_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli",
       "mitoses", "class"]

df=pd.read_csv(url, names=names)

# preprocessing data
df.replace("?", -99999, inplace=True)
# print(df.axes)

df.drop(["id"], 1, inplace=True)

# print the shape of the dataset
# print(df.shape)

# dataset visualization
# print(df.loc[698])
# print(df.describe())

# plot histograms for each variables
# df.hist(figsize=(10,10))
# plt.show()

# create scatter plot matrix
# scatter_matrix(df, figsize=(18,18))
# plt.show()

# create X and Y datasets for training

X = np.array(df.drop(["class"],1))
y = np.array(df["class"])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# specify testing options
seed = 8
scoring = "accuracy"

# define the models to train
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train,
                                                 cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    message = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(message)

# Making predictions on validation dataset

for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

classification =  SVC()

classification.fit(X_train, y_train)
accuracy = classification.score(X_test, y_test)
print(accuracy)

example = np.array([[4,2,1,1,1,2,3,2,10]])
example = example.reshape(len(example), -1)
prediction = classification.predict(example)
print(prediction)