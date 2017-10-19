# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause


from __future__ import division
import time

import numpy as np
import pandas
import csv
import itertools
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

"""Read in dataset"""
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Instance","Feature 1","Feature 2", "Feature 3","Feature 4","Feature 5","Feature 6","Feature 7",
                "Feature 8","Feature 9","Feature 10","Target","Target Class"]
dataframe = pandas.read_csv("C:\Users\bboyd\Documents\college - 4th year\Machine Learning\machine_learning_group_project\team_13\datasets\sum_without_noise.csv",
                             sep=';',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,6,7,8,9,10,11],
                             nrows = 100)
data = np.array(dataframe.as_matrix())
X = data[:,:8]
y = data[:,9]

print("Beginning to fit model")
svr_poly = SVR(kernel='poly', C=1e3,verbose=True)
y_lin = svr_poly.fit(X,y)
print("Model Fit")
lw = 2
plt.scatter(X, y, color='darkorange', label='data')

plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')

plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
