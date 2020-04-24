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
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import pyplot
from numpy.linalg import eig
# from sklearn import mixture
# from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
from pandas import read_csv
from numpy import array
from numpy import mean
from numpy import cov
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


################################################
# PART 1 - K-Means
################################################
print('############### PART 1 ###############')

# kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
# kmeans.labels_
# kmeans.predict([[0, 0,0,0], [12, 3, 8, 1]])
# kmeans.cluster_centers_


# Parameters
k = 3
n_init = 1


numLoops = 0
numIteration = 100
model = KMeans(n_clusters=k, n_init=n_init, max_iter=1)
bestReError = 10000

for i in range(0, numIteration):
    numLoops = i
    curReError = 0
    j = 0

    dis = model.fit_transform(x)
    for label in model.labels_:
        curReError += dis[j][label]
        j += 1
    

    print("Reconstruction error = " + str(curReError))
    if(curReError < bestReError ):
        print("Iteration - " + str(i))

        if( bestReError / curReError - 1 < .01):
            print("Less than 1% improved")
            bestReError = curReError
            break
        else:
            bestReError = curReError
print("--------------------\n\n")
n_init = numLoops

starting_K = 2
ending_K = 20
listOfErrors = []
listOfK = []
bestReError = 10000

for k in range(starting_K, ending_K + 1):
    model = KMeans(n_clusters=k, n_init=n_init)
    curReError = 0
    j = 0

    dis = model.fit_transform(x)
    for label in model.labels_:
        curReError += dis[j][label]
        j += 1
    
    listOfErrors.append(curReError)
    listOfK.append(k)

#plot
plt.plot(listOfK,listOfErrors)
plt.xlabel('K')
plt.ylabel('Reconstruction Error')

#Manually find elbow of the curve
elbow_k = 7 # got this with my eyes
k = elbow_k

#clean data
newY = []
for cur in y:
    if(cur == 'Iris-setosa'):
        newY.append(0)
    if(cur == 'Iris-versicolor'):
        newY.append(1)
    if(cur == 'Iris-virginica'):
        newY.append(2)

# Make prediction with k = elbow_k
model = KMeans(n_clusters=k, n_init=n_init)
prediction = model.fit_predict(x)
# val = np.concatenate((y_two, y_one))
val = y

# Evaluate prediction
print("Scores where K = elbow_k")
print("Accuracy metric can't be calculated K != 3")
# print(accuracy_score(val, prediction))
print("\nConfusion matrix")
# print(confusion_matrix(val, prediction))


# Make prediction with k = 3
k = 3
model = KMeans(n_clusters=k, n_init=n_init, max_iter=1)
prediction = model.fit_predict(x)
val = newY

# Evaluate prediction
print("Scores where K = 3")
print("Accuracy metric")
print(accuracy_score(val, prediction))
print("\nConfusion matrix")
print(confusion_matrix(val, prediction))


# ################################################
# #PART 2 - GMM
# ################################################
print('############### PART 2 ###############')
#Parameters
n_components = 3
n_init = 1

# numLoops = 0
# numIteration = 100
# for x in range(0, numIteration):
#     numLoops = x
#     #run GMM?
#     lower_bound_attribute = 0
#     print("lower_bound_attribute = " + str(lower_bound_attribute))

#     if("Better clustering based on lower_bound_ attribute"):
#         print("Iteration - " + str(x))

#     if("difference between successive better lower_bound_attribute less than 1%"):
#         break

# n_init = numLoops

###### AIC
starting_K = 2
ending_K = 20
listOfAIC = []
listOfK = []

for k in range(starting_K, ending_K + 1):
    model = GaussianMixture(n_components=k)
    AIC = model.fit(x).aic(x)
    listOfAIC.append(AIC)
    listOfK.append(k)

#plot AIC
plt.plot(listOfK,listOfAIC)
plt.xlabel('K')
plt.ylabel('AIC')

#Manually find elbow of the curve
aic_elbow_k = 13 #"Find elbow of curve"

###### BIC

starting_K = 2
ending_K = 20
listOfBIC = []

for k in range(starting_K, ending_K + 1):
    model = GaussianMixture(n_components=k)
    BIC = model.fit(x).bic(x)
    listOfBIC.append(BIC)

#plot BIC
plt.plot(listOfK,listOfBIC)
plt.xlabel('K')
plt.ylabel('BIC')
# plt.show()

#Manually find elbow of the curve
bic_elbow_k = 9 #"Find elbow of curve"


#clean data
newY = []
for cur in y:
    if(cur == 'Iris-setosa'):
        newY.append(0)
    if(cur == 'Iris-versicolor'):
        newY.append(1)
    if(cur == 'Iris-virginica'):
        newY.append(2)


##### Print K = AIC prediction
print("K = AIC elbow")
k = aic_elbow_k

# Make prediction on validation dataset

model = GaussianMixture(n_components=k)
prediction = model.fit(x).predict(x)
val = y

# Evaluate prediction
print("Accuracy metric can't be calculated K != 3")
# print(accuracy_score(val, prediction))
print("\nConfusion matrix")
# print(confusion_matrix(val, prediction))


######## Print K = BIC prediction
print("K = BIC elbow")
k = bic_elbow_k

# Make prediction on validation dataset
model = GaussianMixture(n_components=k)
prediction = model.fit(x).predict(x)
val = y

# Evaluate prediction
print("Accuracy metric can't be calculated K != 3")
# print(accuracy_score(val, prediction))
print("\nConfusion matrix")
# print(confusion_matrix(val, prediction))


######## Print K = 3 prediction
print("K = 3")

k = 3
model = GaussianMixture(n_components=k)
prediction = model.fit(x).predict(x)
val = newY

# Evaluate prediction
print("Accuracy metric")
print(accuracy_score(val, prediction))
print("\nConfusion matrix")
print(confusion_matrix(val, prediction))
