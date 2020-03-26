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

# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

# training model
twoFold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
result = cross_val_score(DecisionTreeClassifier(
), X_train, Y_train, cv=twoFold, scoring='accuracy')

print('%s: %f (%f)' % ('Decision Tree', result.mean(), result.std()))

# Make predictions on validation dataset
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#############################################
# Part 2
#############################################
print("\n\n//////////////////  PART 2 ///////////////\n\n")

# define a matrix
A = X
# calculate the mean of each column
M = mean(A.T, axis=1)
# center columns by subtracting column means
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

# Split-out validation dataset
array = dataset.values
X = array[:, 0:3]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

# training model
twoFold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
result = cross_val_score(DecisionTreeClassifier(
), X_train, Y_train, cv=twoFold, scoring='accuracy')

print('%s: %f (%f)' % ('Decision Tree', result.mean(), result.std()))

# Make predictions on validation dataset
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
