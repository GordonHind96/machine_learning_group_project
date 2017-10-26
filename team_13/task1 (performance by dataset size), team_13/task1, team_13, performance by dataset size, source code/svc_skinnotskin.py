#SVC SKIN NOT SKIN
from __future__ import division
import time

import numpy as np
import pandas as pd
import csv
import itertools
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.classification import accuracy_score,precision_score
from sklearn.model_selection import KFold, GridSearchCV,cross_val_predict
import matplotlib.pyplot as plt

set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Feature 1","Feature 2", "Feature 3","Target"]

i = set_sizes[4]
dataframe = pd.read_csv("C:\\Users\\gordo\\Desktop\\ML\\machine_learning_group_project\\team_13\\datasets\\skin.csv",
                             sep=',',header=0,names=column_names,usecols=[0,1,2,3],
                             nrows =i)
                             	#nrows = set_sizes[4]
dataframe = dataframe.sample(frac=1)
dataframe = dataframe.sample(i)
X_train = dataframe.head(int(i * .7)) 
#X_train.append(dataframe.tail(int(nrows2*.35)))
Y_train = X_train.Target
X_train = X_train[["Feature 1","Feature 2","Feature 3"]]
X_test = dataframe.head(int(i * .3))
#X_test.append(dataframe.tail(int(nrows2 * .15)))

Y_test = X_test.Target
X_test = X_test[["Feature 1", "Feature 2","Feature 3"]]

svc = SVC()
svc.fit(X_train,Y_train)
pred = svc.predict(X_test)
#cv_score = cross_val_predict(svc, X_train, Y_train, cv=10)
print("accuracy_score: %f" % (accuracy_score(Y_test,pred)))
print("precision_score: %f" % (precision_score(Y_test,pred)))
#print(cv_score)