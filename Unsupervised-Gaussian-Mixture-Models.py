# Load libraries
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot
from numpy.linalg import eig
from pandas import read_csv
from numpy import array
from numpy import mean
from numpy import cov

import numpy as np
import csv

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
x_one, x_two, y_one, y_two = train_test_split(
    x, y, test_size=0.50, random_state=1)

twoFold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)

X = np.array([[1, 2], [1, 4], [1, 0],
...               [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
kmeans.labels_
# array([1, 1, 1, 0, 0, 0], dtype=int32)
kmeans.predict([[0, 0], [12, 3]])
# array([1, 0], dtype=int32)
kmeans.cluster_centers_
# array([[10.,  2.],
    #    [ 1.,  2.]])








# modelData = DecisionTreeClassifier()
# # Make prediction on validation dataset
# modelData.fit(x_one, y_one)
# prediction1 = modelData.predict(x_two)

# modelData.fit(x_two, y_two)
# prediction2 = modelData.predict(x_one)

# # combined both predictions
# prediction = np.concatenate((prediction1, prediction2))
# val = np.concatenate((y_two, y_one))
# # prediction = tempPrediction

# # Evaluate prediction
# print("Accuracy metric")
# print(accuracy_score(val, prediction))
# print("\nConfusion matrix")
# print(confusion_matrix(val, prediction))
