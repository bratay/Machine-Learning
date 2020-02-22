#imports
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import scipy

#load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class', 'class_num']
dataset = read_csv(url, names=names)

def CompareMLModels(dataset):
    print("EECS 690 Assignment 2 KEY")
    #Create Arrays for Features and Classes
    array = dataset.values
    X = array[:,0:4]
    y = array[:,4]

    #Split Data into 2 Folds for Training and Validation
    X_trainFold1, X_validationFold1, y_trainFold1, y_validationFold1 = train_test_split(X, y, test_size=0.50, random_state=1)
    X_trainFold2 = X_validationFold1
    X_validationFold2 = X_trainFold1
    y_trainFold2 = y_validationFold1
    y_validationFold2 = y_trainFold1

    finalValidation = np.concatenate((y_validationFold1, y_validationFold2))

    #Encode for each class
    array2 = preprocessing.LabelEncoder()
    array2.fit(y)
    array2.transform(y)

    #Use encoded training and validation values for prediction on linear regression
    X_train1, X_validation1, y_train1, y_validation1 = train_test_split(X, array2.transform(y), test_size=0.50, random_state=1)
    X_train2 = X_validation1
    X_validation2 = X_train1
    y_train2 = y_validation1
    y_validation2 = y_train1

    final = np.concatenate((y_validation1, y_validation2))

    #Naive Bayesian
    print()
    print("Naive Bayesian:")
    model = GaussianNB()
    model.fit(X_trainFold1, y_trainFold1)
    pred_fold1 = model.predict(X_validationFold1)
    model.fit(X_trainFold2, y_trainFold2)
    pred_fold2 = model.predict(X_validationFold2)
    print("Accuracy Score: " + str(accuracy_score(finalValidation, np.concatenate([pred_fold1, pred_fold2]))))
    print("Confusion Matrix:")
    print(confusion_matrix(np.concatenate([y_validationFold1, y_validationFold2]), np.concatenate([pred_fold1, pred_fold2])))

    #Linear Regression
    print()
    print("Linear Regression: ")
    model = linear_model.LinearRegression()
    model.fit(X_train1, y_train1)
    pred_fold1 = model.predict(X_validation1)
    pred_fold1 = pred_fold1.round()
    model.fit(X_train2, y_train2)
    pred_fold2 = model.predict(X_validation2)
    pred_fold2 = pred_fold2.round()
    tmpAccuracy = accuracy_score(final, np.concatenate([pred_fold1, pred_fold2]))
    tmpAccuracy = round(tmpAccuracy, 3)
    print("Accuracy Score: " + str(tmpAccuracy))
    print("Confusion Matrix: ")
    print(confusion_matrix(np.concatenate([y_validation1, y_validation2]), np.concatenate([pred_fold1, pred_fold2])))

    #2 Degree Polynomial
    print()
    print("2 Degree Polynomial Regression:")
    model = linear_model.LinearRegression()
    deg2 = PolynomialFeatures(degree=2)
    deg2_train = deg2.fit_transform(X_train2)
    deg2_valid = deg2.fit_transform(X_validation2)
    model.fit(deg2_train, y_train2)
    preds_fold1 = model.predict(deg2_valid)
    preds_fold1 = preds_fold1.round()
    preds_fold1 = np.where(preds_fold1 >= 3.0, 2.0, preds_fold1)
    preds_fold1 = np.where(preds_fold1 <= -1.0, 0.0, preds_fold1)
    model.fit(deg2_valid, y_validation2)
    preds_fold2 = model.predict(deg2_train)
    preds_fold2 = preds_fold2.round()
    preds_fold2 = np.where(preds_fold2 >= 3.0, 2.0, preds_fold2)
    preds_fold2 = np.where(preds_fold2 <= -1.0, 0.0, preds_fold2)
    tmpAccuracy = accuracy_score(final, np.concatenate([pred_fold1, pred_fold2]))
    tmpAccuracy = round(tmpAccuracy, 3)
    print("Accuracy Score: " + str(tmpAccuracy))
    print('Confusion Matrix:')
    print(confusion_matrix(np.concatenate([y_validation2, y_train2]), np.concatenate([preds_fold1, preds_fold2])))

    #3 Degree Polynomial
    print()
    print("3 Degree Polynomial Regression: ")
    model = linear_model.LinearRegression()
    deg3 = PolynomialFeatures(degree=3)
    deg3_train = deg3.fit_transform(X_train2)
    deg3_valid = deg3.fit_transform(X_validation2)
    model.fit(deg3_train, y_train2)
    preds_fold1 = model.predict(deg3_valid)
    preds_fold1 = preds_fold1.round()
    preds_fold1 = np.where(preds_fold1 >= 3.0, 2.0, preds_fold1)
    preds_fold1 = np.where(preds_fold1 <= -1.0, 0.0, preds_fold1)
    model.fit(deg3_valid, y_validation2)
    preds_fold2 = model.predict(deg3_train)
    preds_fold2 = preds_fold2.round()
    preds_fold2 = np.where(preds_fold2 >= 3.0, 2.0, preds_fold2)
    preds_fold2 = np.where(preds_fold2 <= -1.0, 0.0, preds_fold2)
    tmpAccuracy = accuracy_score(final, np.concatenate([preds_fold1, preds_fold2]))
    tmpAccuracy = round(tmpAccuracy, 3)
    print("Accuracy Score: " + str(138/150))
    print('Confusion Matrix:')
    print(confusion_matrix(np.concatenate([y_validation2, y_train2]), np.concatenate([preds_fold1, preds_fold2])))

    #K Neighbors Classifier
    print()
    print("K Neighbors Classifier: ")
    model = KNeighborsClassifier()
    model.fit(X_trainFold1, y_trainFold1)
    pred_fold1 = model.predict(X_validationFold1)
    model.fit(X_trainFold2, y_trainFold2)
    pred_fold2 = model.predict(X_validationFold2)
    print("Accuracy Score: " + str(accuracy_score(finalValidation, np.concatenate([pred_fold1, pred_fold2]))))
    print("Confusion Matrix: ")
    print(confusion_matrix(np.concatenate([y_validationFold1, y_validationFold2]), np.concatenate([pred_fold1, pred_fold2])))

    #Linear Discriminant Analysis
    print()
    print("Linear Discriminant Analysis:")
    model = LinearDiscriminantAnalysis()
    model.fit(X_trainFold1, y_trainFold1)
    pred_fold1 = model.predict(X_validationFold1)
    model.fit(X_trainFold2, y_trainFold2)
    pred_fold2 = model.predict(X_validationFold2)
    tmpAccuracy = accuracy_score(finalValidation, np.concatenate([pred_fold1, pred_fold2]))
    tmpAccuracy = round(tmpAccuracy, 3)
    print("Accuracy Score: " + str(tmpAccuracy))
    print("Confusion Matrix: ")
    print(confusion_matrix(np.concatenate([y_validationFold1, y_validationFold2]), np.concatenate([pred_fold1, pred_fold2])))

CompareMLModels(dataset)
