# Load libraries
import numpy as np

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
from sklearn import svm
from sklearn.neural_network import MLPClassifier


def printLinearResults(matrix, acc, modelName):
    print("#######################################")
    print("%s" % modelName)
    print("#######################################")

    # Evaluate prediction
    print("Accuracy metric")
    print(acc)
    print("\nConfusion matrix")
    print(matrix)
    print("\n")


def printResults(modelData, modelName):
    print("#######################################")
    print("%s" % modelName)
    print("#######################################")

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
# SVN (Support Vector Machine)
models.append(('Support Vector Machine',  svm.LinearSVC()))
# NN (Neural Network)
models.append(('Neural Network', MLPClassifier()))
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

    printResults(model, name)


# Encode for each class
array2 = preprocessing.LabelEncoder()
array2.fit(y)
array2.transform(y)

# linear regression
X_train1, X_validation1, y_train1, y_validation1 = train_test_split(
    x, array2.transform(y), test_size=0.50, random_state=1)
X_train2 = X_validation1
X_validation2 = X_train1
y_train2 = y_validation1
y_validation2 = y_train1

final = np.concatenate((y_validation1, y_validation2))


# Linear regression (LinearRegression)
model = linear_model.LinearRegression()
model.fit(X_train1, y_train1)
pred_fold1 = model.predict(X_validation1)
pred_fold1 = pred_fold1.round()
model.fit(X_train2, y_train2)
pred_fold2 = model.predict(X_validation2)
pred_fold2 = pred_fold2.round()
tmpAccuracy = accuracy_score(final, np.concatenate([pred_fold1, pred_fold2]))
matrix = confusion_matrix(np.concatenate(
    [y_validation1, y_validation2]), np.concatenate([pred_fold1, pred_fold2]))

printLinearResults(matrix, str(round(tmpAccuracy, 3)), "Linear Regression: ")

array = dataset.values
X = array[:, 0:4]
y = array[:, 4]

X_trainFold1, X_validationFold1, y_trainFold1, y_validationFold1 = train_test_split(
    X, y, test_size=0.50, random_state=1)
X_trainFold2 = X_validationFold1
X_validationFold2 = X_trainFold1
y_trainFold2 = y_validationFold1
y_validationFold2 = y_trainFold1

finalValidation = np.concatenate((y_validationFold1, y_validationFold2))

array2 = preprocessing.LabelEncoder()
array2.fit(y)
array2.transform(y)

X_train1, X_validation1, y_train1, y_validation1 = train_test_split(
    X, array2.transform(y), test_size=0.50, random_state=1)
X_train2 = X_validation1
X_validation2 = X_train1
y_train2 = y_validation1
y_validation2 = y_train1

final = np.concatenate((y_validation1, y_validation2))

# 2 Degree Polynomial
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
tmpAccuracy = accuracy_score(
    final, np.concatenate([pred_fold1, pred_fold2]))
tmpAccuracy = round(tmpAccuracy, 3)

print("#######################################")
print("2 Degree Polynomial Regression")
print("#######################################")

# Evaluate prediction
print("Accuracy metric: " + str(tmpAccuracy))
print("\nConfusion matrix")
print(confusion_matrix(np.concatenate(
    [y_validation2, y_train2]), np.concatenate([preds_fold1, preds_fold2])))
print("\n")

# 3 Degree Polynomial
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
tmpAccuracy = accuracy_score(
    final, np.concatenate([preds_fold1, preds_fold2]))
tmpAccuracy = round(tmpAccuracy, 3)

print("#######################################")
print("3 Degree Polynomial Regression")
print("#######################################")

# Evaluate prediction
print("Accuracy Score: " + str(tmpAccuracy))
print('Confusion Matrix:')
print(confusion_matrix(np.concatenate(
    [y_validation2, y_train2]), np.concatenate([preds_fold1, preds_fold2])))
print("\n")
