import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets, linear_model
from sklearn.metrics import accuracy_score, precision_score, average_precision_score
from sklearn.model_selection import KFold

files="skin.csv"
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Feature 1","Feature 2", "Feature 3","Target"]

nrows2 = set_sizes[8]
dataframe = pd.read_csv(files,
                             sep=',',header=0,names=column_names,usecols=[0,1,2,3],
                             nrows =nrows2)



X_train = dataframe.head(int(nrows2 * .7))
Y_train = X_train.Target
X_train = X_train[["Feature 1","Feature 2","Feature 3"]]
X_test = dataframe.tail(int(nrows2 * .3))
Y_test = X_test.Target
X_test = X_test[["Feature 1", "Feature 2","Feature 3"]]

X_fold = dataframe.head(int(nrows2))
kf = KFold(n_splits=10)
kf.get_n_splits(X_fold)
Y_fold=X_fold.Target
X_fold = X_fold[["Feature 1","Feature 2","Feature 3"]]
for k, (train, test) in enumerate(k_fold.split(X_fold, Y_fold)):
    
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