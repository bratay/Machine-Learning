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

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

array = dataset.values
X = [:, 0:4]
y = [:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.50, random_state=1)

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
# Naïve Baysian (NBClassifier)
models.append(('NBClassifier', GaussianNB()))  # Same as NBClassifier?

results = []
names = []
for name, model in models:
    twoFold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    # train, test, and get result


# print
