import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

file="The SUM dataset, with noise.csv"
properfile="sum_without_noise.csv"
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Instance","Feature 1","Feature 2", "Feature 3","Feature 4","Feature 5","Feature 6","Feature 7",
                "Feature 8","Feature 9","Feature 10","Target","Target Class"]

nrows2 = set_sizes[10]
dataframe = pd.read_csv(properfile,
                             sep=';',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,9,10,11],
                             nrows =nrows2)



X_train = dataframe.head(int(nrows2 * .7))
Y_train = X_train.Target
X_train = X_train[["Feature 1","Feature 2","Feature 3","Feature 4","Feature 5","Feature 6","Feature 7",
                "Feature 8","Feature 9","Feature 10"]]
X_test = dataframe.tail(int(nrows2 * .3))
Y_test = X_test.Target
X_test = X_test[["Feature 1", "Feature 2","Feature 3","Feature 4","Feature 5","Feature 6","Feature 7",
                "Feature 8","Feature 9", "Feature 10"]]


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
