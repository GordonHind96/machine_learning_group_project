from __future__ import division
import time

import numpy as np
import pandas
import csv
import itertools
from sklearn.svm import SVR
from datetime import datetime
from dateutil.parser import parse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.regression import mean_squared_error,r2_score
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score
import matplotlib.pyplot as plt

set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

column_names = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude"
				,"dropoff_longitude","dropoff_latitude","store_and_fwd_flag","trip_duration"]
"""Read in dataset"""
for i in range(0,7):
	dataframe = pandas.read_csv("C:\\Users\\gordo\\Desktop\\ML\\datasets\\train.csv",
                             sep=',',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,10] ,nrows =set_sizes[i])

	Y = dataframe["trip_duration"]
	X = dataframe[["vendor_id","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]

	x_train = X.head(int(set_sizes[i]*0.7))
	x_test = X.tail(int(set_sizes[i]*0.3))

	y_train = Y.head(int(set_sizes[i]*0.7))
	y_test = Y.tail(int(set_sizes[i]*0.3))

# Set up possible values of parameters to optimize over
	svr = SVR()
	svr.fit(x_train.values.reshape(len(x_train), 8),y_train)
	pred = svr.predict(x_test.values.reshape(len(x_test),8))
	print(set_sizes[i])
	print("Mean Squared Error: %f" % (mean_squared_error(y_test,pred)))
	print("R2 Score: %f" % (r2_score(y_test,pred)))