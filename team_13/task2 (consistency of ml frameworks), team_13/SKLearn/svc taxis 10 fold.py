import time

import numpy as np
import pandas as pd
import csv
import itertools
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.classification import accuracy_score,precision_score
from sklearn.model_selection import KFold, GridSearchCV,cross_val_predict
import matplotlib.pyplot as plt

set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

column_names = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude"
				,"dropoff_longitude","dropoff_latitude","store_and_fwd_flag","trip_duration", "Short_or_long"]
"""Read in dataset"""
i=7
dataframe = pd.read_csv("train.csv",
                             sep=',',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,10,11] ,nrows =set_sizes[i])

Y = dataframe["Short_or_long"]
X = dataframe[["vendor_id","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]
svc = SVC()

    
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics.classification import accuracy_score,precision_score

predicted = cross_val_predict(svc, X, Y, cv=10)
print(accuracy_score(Y, predicted))
#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(regr, X, Y, cv=5)
print("Mean squared error: %.2f"
      % mean_squared_error(Y, predicted))