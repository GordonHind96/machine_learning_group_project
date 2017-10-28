from mlxtend.regressor import LinearRegression
#from mlxtend.evaluate import scoring
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict as cvp
from sklearn import linear_model

files = "C:\\Users\\bboyd\\Documents\\college - 4th year\\Machine Learning\\machine_learning_group_project\\team_13\\datasets\\skin.csv"
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Feature 1","Feature 2", "Feature 3","Target"]

nrows = 6
data_size = set_sizes[nrows]
dataframe = pd.read_csv(files,
                             sep=',',header=0,names=column_names,usecols=[0,1,2,3],
                             nrows =data_size)

X = dataframe[["Feature 1","Feature 2","Feature 3"]]
Y = dataframe ["Target"]

print(set_sizes[0])
print('here', set_sizes[nrows]*0.7)

X_train = X.head(int(set_sizes[nrows]*0.7))
X_test = X.tail(int(set_sizes[nrows]*0.3))

Y_train = Y.head(int(set_sizes[nrows]*0.7))
Y_test = Y.tail(int(set_sizes[nrows]*0.3))


ne_lr = LinearRegression(minibatches=None)
Y2 = pd.to_numeric(Y, downcast='float')
print("here",type ((Y2)))

print(type(Y_train))

ne_lr.fit(X_train, pd.to_numeric(Y_train, downcast='float'))

print(ne_lr)

y_pred = ne_lr.predict(X_test)


res = mean_squared_error(Y_test,y_pred)
#res = scoring(y_target=Y_test, y_predicted=y_pred, metric='rmse')
print("results: ", res)


lin = linear_model.LinearRegression()

lin.fit(X_train, Y_train)

predictedCV = cvp(lin,X,Y,cv=10)
print("rmse cross val" , mean_squared_error(Y,predictedCV))