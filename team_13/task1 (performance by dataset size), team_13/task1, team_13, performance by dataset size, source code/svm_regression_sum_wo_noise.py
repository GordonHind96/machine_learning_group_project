
from __future__ import division
import time

import numpy as np
import pandas
import csv
import itertools
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics.regression import mean_squared_error
import matplotlib.pyplot as plt


"""Read in dataset"""
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Instance","Feature 1","Feature 2", "Feature 3","Feature 4","Feature 5","Feature 6","Feature 7",
                "Feature 8","Feature 9","Feature 10","Target","TargetClass"]
dataframe = pandas.read_csv("C:\\Users\\gordo\\Desktop\\ML\\datasets\\without-noise\\The-SUM-dataset-without-noise.csv",
                             sep=';',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,9,10,11],
                             nrows =set_sizes[4])
train_size = int(set_sizes[4] * .7)
test_size = int(set_sizes[4] * .3)

X_train = dataframe.head(train_size)
Y_train = X_train.Target
X_train = X_train[["Feature 1","Feature 2","Feature 3","Feature 4","Feature 6","Feature 7","Feature 8","Feature 9","Feature 10"]]
X_test = dataframe.tail(test_size)
Y_test = X_test.Target
X_test = X_test[["Feature 1","Feature 2","Feature 3","Feature 4","Feature 6","Feature 7","Feature 8","Feature 9","Feature 10"]]
test_se = 0

nScore =-0.9
oldScore = -1
cVal = 1

pred_test = dataframe
while(not(nScore <= oldScore)):
    print(cVal)
    svr = SVR(C=cVal,epsilon=0.0000001,verbose=True).fit(X_train,Y_train)
    pred_test = svr.predict(X_test)
    score = svr.score(X_test,Y_test)
    test_se = mean_squared_error(Y_test, pred_test)
    print(score)
    print(test_se)


    cVal = cVal * 2
    oldScore = nScore
    nScore = score


print("Final Score: %f" % nScore)
print("Final SE: %f" %test_se)
print("Final C: %d" %cVal)


