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

import numpy as np
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

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
printResults(DecisionTreeClassifier())

#############################################
# Part 2 PCA
#############################################
print("\n\n//////////////////  PART 2 //////////////////\n\n")

# define a matrix
A = x
# calculate the mean of each column and center columns
M = mean(A.T, axis=1)
C = A - M
# calculate covariance matrix of centered matrix
V = cov(C.T.astype(float))
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print("/////////////" + " eigen values " + "/////////////")
print(vectors)
print("\n/////////////" + " eigen vectors " + "/////////////")
print(values)
print('\nPoV = ' + str(values[0] / (values[0] + values[1] + values[2])) + "\n")

# For part 3 and 4
P = vectors.T.dot(C.T)
newFeatures = P.T

# used in part 4
transformedFeatures = vectors.T.dot(C.T)

# Split-out validation dataset
array = dataset.values
x = array[:, 0:3]
y = array[:, 4]
x_one, x_two, y_one, y_two = train_test_split(
    x, y, test_size=0.50, random_state=1)

twoFold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
printResults(DecisionTreeClassifier())

#############################################
# Part 3 simulated annealing
#############################################
print("\n\n////////////////// PART 3 //////////////////\n\n")

array = dataset.values
x = array[:, 0:4]
y = array[:, 4]

allData8F = np.concatenate((newFeatures, x), axis=1)
# print(allData8F)

X = allData8F[:, 0:8]
print(X)

curPerformance = 0
prePreformance = 0
for x in range(0, 100):
    # fit model
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=0.50, random_state=1)
    curPerformance = 1  # estimate performance
    if(curPerformance > prePreformance):
        # print("ACCEPT")
        prePreformance = curPerformance
    else:
        acceptProbability = 0
        ranUnform = 0 ########### I dont know what you mean by random uniform

        if(ranUnform > acceptProbability):
            # print("REJECT")
            curPerformance = 0
        else:
            # print("ACCEPT")
            prePreformance = curPerformance

#############################################
# Part 4 genetic algorithm
#############################################
print("\n\n////////////////// PART 4 //////////////////\n\n")

