
from __future__ import division
import time

import numpy as np
import pandas
import csv
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score,precision_score
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier


"""Read in dataset"""

set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
nrows2 = set_sizes[0]
column_names = ["Instance","Feature 1","Feature 2", "Feature 3","Feature 4","Feature 5","Feature 6","Feature 7",
                "Feature 8","Feature 9","Feature 10","Target","TargetClass"]
dataframe = pandas.read_csv("C:\\Users\\gordo\\Desktop\\ML\\datasets\\with-noise\\The-SUM-dataset-with-noise.csv",
                             sep=';',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,9,10,12],
                             nrows =nrows2)

X_train = dataframe.head(int(nrows2 * .7))
Y_train = X_train.TargetClass
X_train = X_train[["Feature 1","Feature 2","Feature 3","Feature 4","Feature 5","Feature 6","Feature 7",
                "Feature 8","Feature 9","Feature 10"]]
X_test = dataframe.tail(int(nrows2 * .3))
Y_test = X_test.TargetClass
X_test = X_test[["Feature 1", "Feature 2","Feature 3","Feature 4","Feature 5","Feature 6","Feature 7",
                "Feature 8","Feature 9", "Feature 10"]]
print("Creating Classifier")
svc = OneVsRestClassifier(SVC(C= 100,kernel='linear'))
print("Fitting data")
svc.fit(X_train,Y_train)
print("Predicting")
pred = svc.predict(X_test)
##cv_score = cross_val_predict(svc, X_train, Y_train, cv=10)
#for i in range(0,100):
	##print(cv_score)print("Y: %s and P: %s"% (Y_test.iloc[i],pred[i]))
print("accuracy_score: %f" % (accuracy_score(Y_test,pred)))
print("precision_score: %f" % (precision_score(Y_test,pred,average='micro')))
