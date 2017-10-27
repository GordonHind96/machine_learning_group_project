import time

import numpy as np
import pandas as pd
import csv
import itertools
from sklearn import linear_model, datasets, linear_model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.classification import accuracy_score,precision_score
from sklearn.model_selection import KFold, GridSearchCV,cross_val_predict
import matplotlib.pyplot as plt

set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

column_names = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude"
				,"dropoff_longitude","dropoff_latitude","store_and_fwd_flag","trip_duration", "Short_or_long"]
"""Read in dataset"""
i=8
dataframe = pd.read_csv("train.csv",
                             sep=',',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,10,11] ,nrows =set_sizes[i])

Y = dataframe["Short_or_long"]
X = dataframe[["vendor_id","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]

X_train = X.head(int(set_sizes[i]*0.7))
X_test = X.tail(int(set_sizes[i]*0.3))

Y_train = Y.head(int(set_sizes[i]*0.7))
Y_test = Y.tail(int(set_sizes[i]*0.3))

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_train, Y_train)


pred = logreg.predict(X_test)

# The coefficients
#print('Coefficients: \n', clf.coef_)
# The mean squared error
t=accuracy_score(Y_test, pred)
tt=precision_score(Y_test, pred, average='weighted')
#print(Y_tester_targets.size)
#print(pred_test.size)
print('Accuracy score: %.2f' % t)
print('Precision: %.2f'% tt)