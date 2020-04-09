# Load libraries
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

print("Class balance accuracy")
matrix = confusion_matrix(val, prediction)

classOne = float(min(matrix[0][0] / float(matrix[0][0] + matrix[0][1] + matrix[0][2] ) , matrix[0][0] / (matrix[0][0] + matrix[1][0] + matrix[2][0] ) ))
classTwo = float(min(matrix[1][1] / float((matrix[1][1] + matrix[1][0] + matrix[1][2] )), matrix[1][1] / float((matrix[1][0] + matrix[1][1] + matrix[1][2] ) )))
classThree = float(min(matrix[2][2] / float((matrix[2][0] + matrix[2][1] + matrix[2][2] ) ), matrix[2][2] / float((matrix[0][2] + matrix[1][2] + matrix[2][2] ) )))

print( (classOne + classTwo + classThree ) / 3 )

print("Balance accuracy")
# print( np.sum( matrix, axis = 0)[2])
# print(float( np.sum( matrix, axis = 0)[2] + np.sum( matrix, axis = 0)[1] - matrix[0][1] - matrix[0][2] ) )
classOne = float(min( float( np.sum( matrix, axis = 0)[2] + np.sum( matrix, axis = 0)[1] - matrix[0][1] - matrix[0][2] ) /  (float( np.sum( matrix, axis = 0)[2] + np.sum( matrix, axis = 0)[1] ) ) , matrix[0][0] / (matrix[0][0] + matrix[1][0] + matrix[2][0] ) ))
classTwo = float(min( float( np.sum( matrix, axis = 0)[2] + np.sum( matrix, axis = 0)[0] - matrix[1][0] - matrix[1][2] ) /  (float( np.sum( matrix, axis = 0)[2] + np.sum( matrix, axis = 0)[0] ) )  , matrix[1][1] / float((matrix[1][0] + matrix[1][1] + matrix[1][2] ) )))
classThree = float(min(float( np.sum( matrix, axis = 0)[0] + np.sum( matrix, axis = 0)[1] - matrix[2][1] - matrix[2][0] ) /  (float( np.sum( matrix, axis = 0)[0] + np.sum( matrix, axis = 0)[1] ) ) , matrix[2][2] / float((matrix[0][2] + matrix[1][2] + matrix[2][2] ) )))

print( (classOne + classTwo + classThree ) / 3 )


#############################################
# Part 2 PCA
#############################################
print("\n\n//////////////////  PART 2 //////////////////\n\n")


#############################################
# Part 3
#############################################
print("\n\n////////////////// PART 3 //////////////////\n\n")

#############################################
# Part 4
#############################################
print("\n\n////////////////// PART 4 //////////////////\n\n")
