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

i = set_sizes[7]
dataframe = pd.read_csv("skin.csv",
                             sep=',',header=0,names=column_names,usecols=[0,1,2,3],
                             nrows =i)

svc = SVC()
X_fold = dataframe.head(i)
Y_fold=X_fold.Target
X_fold = X_fold[["Feature 1","Feature 2","Feature 3"]]
    
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics.classification import accuracy_score,precision_score

predicted = cross_val_predict(svc, X_fold, Y_fold, cv=10)
print(accuracy_score(Y_fold, predicted))
#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(regr, X, Y, cv=5)
print("Mean squared error: %.2f"
      % mean_squared_error(Y_fold, predicted))