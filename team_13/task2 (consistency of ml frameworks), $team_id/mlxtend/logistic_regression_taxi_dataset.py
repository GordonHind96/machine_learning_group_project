from mlxtend.plotting import plot_decision_regions
from mlxtend.classifier import LogisticRegression

import pandas
import numpy as np
from sklearn import linear_model, datasets, linear_model


set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

column_names = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude"
				,"dropoff_longitude","dropoff_latitude","store_and_fwd_flag","trip_duration","Short_or_long"]
"""Read in dataset"""

i=8
dataframe = pandas.read_csv("C:\\Users\\bboyd\\Downloads\\train\\train.csv",
sep=',',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,10,11] ,nrows = set_sizes[i])

dataframe.dropna(axis=0, how='all')

Y = dataframe["Short_or_long"]

X = dataframe[["passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]
X = X.get_values()

print(type(Y))
Y = Y.get_values()

X_train = X[:int(set_sizes[i]*0.7) ]
X_test = X[int(set_sizes[i]*0.3) : ]

len = Y.size;
train_len = (int) ((len*.7))
Y_train = Y[ : train_len ]

test_len = (int) (len - train_len)

Y_test = Y[ train_len : ]

print("len ", len)
print("Y train " , Y_train.size)
print("X train " , (int) (X_train.size/6))

print("Y test  " , Y_test.size)
print("X test  " , (int) (X_test.size/6))

print(type(Y_test))

lr = LogisticRegression(eta=0.05,
                        l2_lambda=0.0,
                        epochs=50,
                        minibatches=1, # for Gradient Descent
                        random_seed=1,
                        print_progress=3)
lr.fit(X_train, Y_train)

print("...")
pre = lr.predict(X_train)


correct = 0
total = 0

i2 = 0
while(i2 < Y_test.size):
    if(Y_test[i2] == pre[i2]):
        correct += 1
    i2+=1
    total += 1

print("--------")
print(pre[0])
print(Y_test[0])

acc = correct/total
print("ACC " , acc)

print("score," , lr.score(X_test,Y_test))
