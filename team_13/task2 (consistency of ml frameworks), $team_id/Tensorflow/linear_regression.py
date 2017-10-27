import tensorflow as tf 
import pandas as pd 
import numpy as np 

column_names = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude"
        ,"dropoff_longitude","dropoff_latitude","store_and_fwd_flag","trip_duration"]
dataframe = pandas.read_csv("C:\\Users\\gordo\\Desktop\\ML\\datasets\\train.csv",
 sep=',',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,10] ,nrows =100000)

Y = dataframe["trip_duration"]
X = dataframe[["vendor_id","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]

x_train = X.head(int(set_sizes[i]*0.7))
x_test = X.tail(int(set_sizes[i]*0.3))

y_train = Y.head(int(set_sizes[i]*0.7))
y_test = Y.tail(int(set_sizes[i]*0.3))