import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, average_precision_score


files="skin.csv"
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Feature 1","Feature 2", "Feature 3","Target"]

nrows2 = set_sizes[8]
dataframe = pd.read_csv(files,
                             sep=',',header=0,names=column_names,usecols=[0,1,2,3],
                             nrows =nrows2)


dataframe = dataframe.sample(frac=1)
dataframe = dataframe.sample(nrows2, replace='True')
X_train = dataframe.head(int(nrows2 * .7))
Y_train = X_train.Target
X_train = X_train[["Feature 1","Feature 2","Feature 3"]]
X_test = dataframe.tail(int(nrows2 * .3))
Y_test = X_test.Target
X_test = X_test[["Feature 1", "Feature 2","Feature 3"]]

h2 = .02  # step size in the mesh
logreg2 = linear_model.LinearRegression()

X_fold = dataframe.head(int(nrows2))
#kf = KFold(n_splits=10)
#kf.get_n_splits(X_fold)
Y_fold=X_fold.Target
i=0


X_fold = X_fold[["Feature 1","Feature 2","Feature 3"]]
#for k, (train, test) in enumerate(kf.split(X_fold, Y_fold)):
#    logreg2.fit(X_fold[:(len(train)-1)],Y_fold[:(len(train)-1)])
#    pred2 =logreg2.predict(X_fold[:(len(test)-1)])
#    #print(pred2)
#    i=1
#    #tt22=accuracy_score(Y_fold[:(len(test)-1)], pred2)
#    #print('Precision: %.2f'% tt22)
#    print("Mean squared error: %.2f"
#      % mean_squared_error(Y_fold[:(len(test)-1)], pred2))
#    print('Variance score: %.2f' % r2_score(Y_fold[:(len(test)-1)], pred2))



regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_tester_pred = regr.predict(X_test)
# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
#print("Mean squared error: %.2f"
#      % mean_squared_error(Y_test, Y_tester_pred))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(Y_test, Y_tester_pred))
#X_test.head()
#Y_test.head()
#X_train.head(10000)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics.classification import accuracy_score,precision_score

predicted = cross_val_predict(regr, X_fold, Y_fold, cv=10)
print(r2_score(Y_fold, predicted))
#clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(logreg2, X_fold, Y_fold, cv=10)
print("Mean squared error: %.2f"
      % mean_squared_error(Y_fold, predicted))