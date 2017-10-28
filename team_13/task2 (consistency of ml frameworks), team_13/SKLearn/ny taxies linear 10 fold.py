import numpy as np
import pandas
import csv
import itertools
from sklearn.svm import SVR
from datetime import datetime
from dateutil.parser import parse
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.regression import mean_squared_error,r2_score
from sklearn.metrics.classification import accuracy_score,precision_score
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score
import matplotlib.pyplot as plt

set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

column_names = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude"
				,"dropoff_longitude","dropoff_latitude","store_and_fwd_flag","trip_duration", "Short_or_long"]
"""Read in dataset"""
i=10
dataframe = pandas.read_csv("train.csv",
                             sep=',',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,10, 11] ,nrows =set_sizes[i])

Y = dataframe["trip_duration"]
X = dataframe[["vendor_id","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]

X_train = X.head(int(set_sizes[i]*0.7))
X_test = X.tail(int(set_sizes[i]*0.3))

Y_train = Y.head(int(set_sizes[i]*0.7))
Y_test = Y.tail(int(set_sizes[i]*0.3))

regr = linear_model.LinearRegression()


h2 = .02  # step size in the mesh
logreg2 = linear_model.LogisticRegression(C=1e5)

X_fold = dataframe.head(int(set_sizes[i]))
kf = KFold(n_splits=10)
kf.get_n_splits(X_fold)
Y_fold=X_fold.Short_or_long
i=0
#print(X_fold.size)
#print(Y_fold.size)
#X_fold = X_fold[["vendor_id","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]
#for k, (train, test) in enumerate(kf.split(X_fold, Y_fold)):
#    logreg2.fit(X_fold[:(len(train)-1)],Y_fold[:(len(train)-1)])
#    pred2 =logreg2.predict(X_fold[:(len(test)-1)])
#    print(pred2)
#    i=1
#    tt22=accuracy_score(Y_fold[:(len(test)-1)], pred2)
#    print('Precision: %.2f'% tt22)
    
    
    
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics.classification import accuracy_score,precision_score

predicted = cross_val_predict(regr, X, Y, cv=10)
print(r2_score(Y, predicted))
#clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(regr, X, Y, cv=5)
print("Mean squared error: %.2f"
      % mean_squared_error(Y, predicted))
