# Load libraries
from collections import Counter
from sklearn.datasets import make_classification
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import csv
import numpy as np
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from sklearn.datasets import make_classification

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids

# Load dataset
dataset = read_csv("imbalanced iris.csv")

#############################################
# Part 1
#############################################
print("\n\n////////////////// PART 1 //////////////////\n\n")


def printResults(modelData):
    # Make prediction on validation dataset
    modelData.fit(x_one, y_one)
    prediction1 = modelData.predict(x_two)

    modelData.fit(x_two, y_two)
    prediction2 = modelData.predict(x_one)

    # combined both predictions
    prediction = np.concatenate((prediction1, prediction2))
    val = np.concatenate((y_two, y_one))
    # prediction = tempPrediction

    # Evaluate prediction
    print("Accuracy metric")
    print(accuracy_score(val, prediction))
    print("\nConfusion matrix")
    print(confusion_matrix(val, prediction))


array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
x_one, x_two, y_one, y_two = train_test_split(
    x, y, test_size=0.50, random_state=1)

twoFold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)

modelData = DecisionTreeClassifier()
# Make prediction on validation dataset
modelData.fit(x_one, y_one)
prediction1 = modelData.predict(x_two)

modelData.fit(x_two, y_two)
prediction2 = modelData.predict(x_one)

# combined both predictions
prediction = np.concatenate((prediction1, prediction2))
val = np.concatenate((y_two, y_one))
# prediction = tempPrediction

# Evaluate prediction
print("Accuracy metric")
print(accuracy_score(val, prediction))
print("\nConfusion matrix")
print(confusion_matrix(val, prediction))

print("\nClass balance accuracy")
matrix = confusion_matrix(val, prediction)

classOne = float(min(matrix[0][0] / float(matrix[0][0] + matrix[0][1] +
                                          matrix[0][2]), matrix[0][0] / (matrix[0][0] + matrix[1][0] + matrix[2][0])))
classTwo = float(min(matrix[1][1] / float((matrix[1][1] + matrix[1][0] + matrix[1][2])),
                     matrix[1][1] / float((matrix[1][0] + matrix[1][1] + matrix[1][2]))))
classThree = float(min(matrix[2][2] / float((matrix[2][0] + matrix[2][1] + matrix[2][2])),
                       matrix[2][2] / float((matrix[0][2] + matrix[1][2] + matrix[2][2]))))

print((classOne + classTwo + classThree) / 3)

print("\nBalance accuracy")
classOne = float(min(float(np.sum(matrix, axis=0)[2] + np.sum(matrix, axis=0)[1] - matrix[0][1] - matrix[0][2]) / (float(
    np.sum(matrix, axis=0)[2] + np.sum(matrix, axis=0)[1])), matrix[0][0] / (matrix[0][0] + matrix[1][0] + matrix[2][0])))
classTwo = float(min(float(np.sum(matrix, axis=0)[2] + np.sum(matrix, axis=0)[0] - matrix[1][0] - matrix[1][2]) / (float(
    np.sum(matrix, axis=0)[2] + np.sum(matrix, axis=0)[0])), matrix[1][1] / float((matrix[1][0] + matrix[1][1] + matrix[1][2]))))
classThree = float(min(float(np.sum(matrix, axis=0)[0] + np.sum(matrix, axis=0)[1] - matrix[2][1] - matrix[2][0]) / (float(
    np.sum(matrix, axis=0)[0] + np.sum(matrix, axis=0)[1])), matrix[2][2] / float((matrix[0][2] + matrix[1][2] + matrix[2][2]))))

print((classOne + classTwo + classThree) / 3)


#############################################
# Part 2 Oversampling
#############################################
print("\n\n//////////////////  PART 2 //////////////////\n\n")


def printSampling(newX, newY):
    x = newX
    y = newY
    x_one, x_two, y_one, y_two = train_test_split(
        x, y, test_size=0.50, random_state=1)

    modelData = DecisionTreeClassifier()
    # Make prediction on validation dataset
    modelData.fit(x_one, y_one)
    prediction1 = modelData.predict(x_two)

    modelData.fit(x_two, y_two)
    prediction2 = modelData.predict(x_one)

    # combined both predictions
    prediction = np.concatenate((prediction1, prediction2))
    val = np.concatenate((y_two, y_one))

    # Evaluate prediction
    print("Accuracy metric")
    print(accuracy_score(val, prediction))
    print("\nConfusion matrix")
    print(confusion_matrix(val, prediction))


X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)

print("-- Random oversampling -- \n")
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)

printSampling(X_resampled, y_resampled)

print("\n\n-- SMOTE -- \n")

X_resampled, y_resampled = SMOTE().fit_resample(X, y)
printSampling(X_resampled, y_resampled)


print("\n\n-- ADASYN -- \n")

X_resampled, y_resampled = ADASYN().fit_resample(X, y)
printSampling(X_resampled, y_resampled)


#############################################
# Part 3 Undersampling
#############################################
print("\n\n////////////////// PART 3 //////////////////\n\n")

X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)


print("-- Random undersampling -- \n")
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)

printSampling(X_resampled, y_resampled)


print("\n\n-- Cluster undersampling -- \n")
cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X, y)

printSampling(X_resampled, y_resampled)

print("\n\n-- Tomek links  -- \n")
tom = TomekLinks()
X_resampled, y_resampled = tom.fit_resample(X, y)

printSampling(X_resampled, y_resampled)

