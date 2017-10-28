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

column_names = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude"
				,"dropoff_longitude","dropoff_latitude","store_and_fwd_flag","trip_duration", "Short_or_long"]
"""Read in dataset"""
i=6
dataframe = pd.read_csv("train.csv",
                             sep=',',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,10,11] ,nrows =set_sizes[i])

Y = dataframe["Short_or_long"]
X = dataframe[["vendor_id","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]

X_train = X.head(int(set_sizes[i]*0.7))
X_test = X.tail(int(set_sizes[i]*0.3))

Y_train = Y.head(int(set_sizes[i]*0.7))
Y_test = Y.tail(int(set_sizes[i]*0.3))

svc = SVC()
svc.fit(X_train,Y_train)
pred = svc.predict(X_test)
#cv_score = cross_val_predict(svc, X_train, Y_train, cv=10)
print("accuracy_score: %f" % (accuracy_score(Y_test,pred)))
print("precision_score: %f" % (precision_score(Y_test,pred,average="micro")))
#print(cv_score)