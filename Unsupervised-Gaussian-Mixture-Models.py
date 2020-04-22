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
import matplotlib.pyplot as plt


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

kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
kmeans.labels_
kmeans.predict([[0, 0,0,0], [12, 3, 8, 1]])
kmeans.cluster_centers_


#Parameters 
k = 3
n_init = 1

numLoops = 0
numIteration = 100
for x in range(0, numIteration):
    numLoops = x
    #run K-means?
    reError = 0
    print("Reconstruction error = " + reError)
    
    if("Better clustering based on reconstruction error"):
        print("Iteration - " + x)
        
    if("difference between successive better reconstruction errors is less than 1%"):
        break

n_init = numLoops

starting_K = 2
ending_K = 20
listOfErrors = []
listOfK = []

for x in range(starting_K, ending_K + 1):
    #run K-means?
    reError = 0
    listOfErrors.append(reError)
    listOfK.append(x)
    
#plot 
plt.plot(listOfK,listOfErrors)
plt.xlabel('K')
plt.ylabel('Reconstruction Error')



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
