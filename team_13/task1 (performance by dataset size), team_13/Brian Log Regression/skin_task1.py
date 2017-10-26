# 245,057 rows

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, average_precision_score

files = "C:\\Users\\bboyd\\Documents\\college - 4th year\\Machine Learning\\machine_learning_group_project\\team_13\\datasets\\skin.csv"
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Feature 1","Feature 2", "Feature 3","Target"]

nrows2 = set_sizes[1]
dataframe = pd.read_csv(files,
                             sep=',',header=0,names=column_names,usecols=[0,1,2,3],
                             nrows =nrows2)


from sklearn.utils import shuffle
dataframe = shuffle(dataframe)

X_train = dataframe.head(int(nrows2 * .7))
Y_train = X_train.Target
X_train = X_train[["Feature 1","Feature 2","Feature 3"]]
X_test = dataframe.tail(int(nrows2 * .3))
Y_test = X_test.Target
X_test = X_test[["Feature 1", "Feature 2","Feature 3"]]





h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_train, Y_train)


pred = logreg.predict(X_test)


accuracy = accuracy_score(Y_test, pred)
precision = precision_score(Y_test, pred)

print('Accuracy score: %.2f' % accuracy)
print('Precision: %.2f'% precision)