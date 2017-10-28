import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets, linear_model
from sklearn.metrics import accuracy_score, precision_score, average_precision_score
from sklearn.model_selection import KFold

files="skin.csv"
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Feature 1","Feature 2", "Feature 3","Target"]

nrows2 = set_sizes[7]
dataframe = pd.read_csv(files,
                             sep=',',header=0,names=column_names,usecols=[0,1,2,3],
                             nrows =nrows2)



X_train = dataframe.head(int(nrows2 * .7))
Y_train = X_train.Target
X_train = X_train[["Feature 1","Feature 2","Feature 3"]]
X_test = dataframe.tail(int(nrows2 * .3))
Y_test = X_test.Target
X_test = X_test[["Feature 1", "Feature 2","Feature 3"]]

h2 = .02  # step size in the mesh
logreg2 = linear_model.LogisticRegression(C=1e5)

X_fold = dataframe.head(int(nrows2))
kf = KFold(n_splits=10)
kf.get_n_splits(X_fold)
Y_fold=X_fold.Target
i=0
#print(X_fold.size)
#print(Y_fold.size)
X_fold = X_fold[["Feature 1","Feature 2","Feature 3"]]
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
from sklearn.metrics.regression import mean_squared_error,r2_score

predicted = cross_val_predict(logreg2, X_fold, Y_fold, cv=10)
print(accuracy_score(Y_fold, predicted))
#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(regr, X_fold, Y_fold, cv=5)
print("Mean squared error: %.2f"
      % mean_squared_error(Y_fold, predicted))