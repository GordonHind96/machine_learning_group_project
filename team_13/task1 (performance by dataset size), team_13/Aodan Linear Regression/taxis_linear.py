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
	dataframe = pandas.read_csv("train.csv",
                             sep=',',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,10] ,nrows =set_sizes[i])

	Y = dataframe["trip_duration"]
	X = dataframe[["vendor_id","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]

	X_train = X.head(int(set_sizes[i]*0.7))
	X_test = X.tail(int(set_sizes[i]*0.3))

	Y_train = Y.head(int(set_sizes[i]*0.7))
	Y_test = Y.tail(int(set_sizes[i]*0.3))

regr = linear_model.LinearRegression()


# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_tester_pred = regr.predict(X_test)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_tester_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_tester_pred))
#X_test.head()
#Y_test.head()
#X_train.head(10000)