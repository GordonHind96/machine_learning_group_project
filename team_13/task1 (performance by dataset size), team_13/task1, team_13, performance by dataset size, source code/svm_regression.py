
from __future__ import division
import time

import numpy as np
import pandas
import csv
import itertools
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.regression import mean_squared_error
import matplotlib.pyplot as plt


"""Read in dataset"""
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Instance","Feature 1","Feature 2", "Feature 3","Feature 4","Feature 5","Feature 6","Feature 7",
                "Feature 8","Feature 9","Feature 10","Target","Target Class"]
dataframe = pandas.read_csv("C:\\Users\\gordo\\Desktop\\ML\\datasets\\without-noise\\The-SUM-dataset-without-noise.csv",
                             sep=';',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,6,7,8,9,10,11],
                             nrows =set_sizes[2])

X_train = dataframe.head(700)
Y_train = X_train.Target
X_train = X_train[["Feature 1","Feature 2", "Feature 3","Feature 4","Feature 6","Feature 7", "Feature 8","Feature 9","Feature 10"]]
X_test = dataframe.tail(300)
Y_test = X_test.Target
X_test = X_test[["Feature 1","Feature 2", "Feature 3","Feature 4","Feature 6","Feature 7", "Feature 8","Feature 9","Feature 10"]]
i = 0
for m in Y_train:
    print("Train Target %f" % (m))
    print(i)
    i = i+1
for m in Y_test:
    print("Test Target %f" % (m))
    print(i)
    i = i+1


print("Creating model")
svr_poly = SVR(kernel='linear', C=1e3)
print("Beginning to fit model ...")
pred_test = svr_poly.fit(X_train,Y_train).predict(X_test)
for i in range (0,300):
    print("Y test %f  Pred test %f" %(Y_test.iloc[i],pred_test[i]))
test_se = mean_squared_error(Y_test, pred_test)

print("Model Fit")
print(test_se)

