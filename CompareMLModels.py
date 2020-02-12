# Load libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

import numpy as np

def printResults(modelData, modelName):
    print("#######################################")
    print("%s" % modelName)
    print("#######################################")
    
    # Make prediction on validation dataset
    modelData.fit(x_one, y_one)
    prediction1 = modelData.predict(x_two)

    modelData.fit(x_two, y_two)
    prediction2 = modelData.predict(x_one)

    #combined both predictions
    prediction = np.concatenate((prediction1, prediction2))
    val = np.concatenate((y_two, y_one))

    # Evaluate prediction
    print("Accuracy metric")
    print(accuracy_score(val, prediction))
    print("\nConfusion matrix")
    print(confusion_matrix(val, prediction))
    print("\n")

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
x_one, x_two, y_one, y_two = train_test_split(
    x, y, test_size=0.50, random_state=1)

models = []
# Linear regression (LinearRegression)
models.append(('LinearRegression', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
# Polynomial of degree 2 regression (LinearRegression)
models.append(('LinearRegression degree 2', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
# Polynomial of degree 3 regression (LinearRegression)
models.append(('LinearRegression degree 3', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
# kNN (KNeighborsClassifier)
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
# LDA (LinearDiscriminantAnalysis)
models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
# Naive Baysian (NBClassifier)
models.append(('NBClassifier', GaussianNB()))  # Same as NBClassifier?

names = []
# Train make prediction and print result
for name, model in models:
    twoFold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)

    cv_results = cross_val_score(
        model, x_one, y_one, cv=twoFold, scoring='accuracy')

    names.append(name)

    printResults(model, name)
