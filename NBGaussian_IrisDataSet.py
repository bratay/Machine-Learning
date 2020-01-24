# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Partition dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.50, random_state=1)

# training model
# 'Gaussian Model', GaussianNB()
tenFoldVal = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
result = cross_val_score( GaussianNB(), X_train, Y_train, cv = tenFoldVal, scoring='accuracy')

print('%s: %f (%f)' % ('Gaussian Model', result.mean(), result.std()))

# Make predictions on validation dataset
model = GaussianNB()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print('\n')

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print('\n')
print(confusion_matrix(Y_validation, predictions))
print('\n')
print(classification_report(Y_validation, predictions))
