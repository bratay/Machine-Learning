# Load libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pandas import read_csv
from numpy import array
from numpy import mean
from numpy import cov
import numpy as np
import csv

########################################
# Policy Iteration RL Algorithm
########################################

# pick Q function
initialQFunc = 0

curQ = 0
policy = 0
nextQ = 0

while():
    # Compute QvT(s,q) (t'th q function)
    curQ = 0
    
    # PIEvt(s) = compute greddy policy
    policy = 0

    # get q+1
    nextQ = 0


########################################
# Value Iteration RL Algorithm
########################################