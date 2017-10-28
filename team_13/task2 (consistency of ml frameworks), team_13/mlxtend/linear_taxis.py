from mlxtend.regressor import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error


set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

column_names = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude"
				,"dropoff_longitude","dropoff_latitude","store_and_fwd_flag","trip_duration","Short_or_long"]
"""Read in dataset"""

data_size = 0
dataframe = pd.read_csv("C:\\Users\\bboyd\\Downloads\\train\\train.csv",
sep=',',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,10,11] ,nrows = set_sizes[data_size])

Y = dataframe["trip_duration"]
X = dataframe[["vendor_id","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]


X_train = X.head(int(set_sizes[data_size]*0.7))
X_test = X.tail(int(set_sizes[data_size]*0.3))

Y_train = Y.head(int(set_sizes[data_size]*0.7))
Y_test = Y.tail(int(set_sizes[data_size]*0.3))


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
