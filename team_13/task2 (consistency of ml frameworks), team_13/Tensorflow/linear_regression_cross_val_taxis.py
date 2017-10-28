import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
rng = np.random
#params
learning_rate = 0.01
training_epochs = 10
display_steps = 50
kfold = KFold(10) #Set up number folds


column_names = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude"
        ,"dropoff_longitude","dropoff_latitude","store_and_fwd_flag","trip_duration"]
"""Read in dataset"""
n_samples = 100000
dataframe = pd.read_csv(tf.gfile.Open("C:\\Users\\gordo\\Desktop\\ML\\datasets\\train.csv"),
sep=',',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,10] ,nrows =n_samples)

_y = dataframe["trip_duration"]
_x = dataframe[["passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]
_x = _x.get_values()
#print(X)


for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices)) # show train test indices for kfold
# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# model weights set to random
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

#Linear model construction
pred = tf.add(tf.multiply(X,W),b)

#Mean Squared Error
cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)

#Gradient Descent
alpha = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Init
init = tf.global_variables_initializer()

#Trainer
print("Start trinaing")
with tf.Session() as sesh:
  sesh.run(init)

  for epoch in range(training_epochs):
    print("in epoch")
    for (x,y) in zip(_x, _y):
      sesh.run(alpha, feed_dict={X: x, Y: y})

      #logs
      if(epoch+1) % display_steps == 0:
         c = sesh.run(cost, feed_dict={X:_x,Y:_y})
         print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),\
          "W=",sess.run(W),"b=", sess.run(b))

  print("FInished")

  plt.plot(_x,_y,'ro')
  plt.plot(_x,sesh.run(W)* X_train + sesh.run(b))
  plt.show()

  #Test
  print("Testing Mean Squared Error")
  #print("X_test shape:",X_test.shape())
  testing_cost = sesh.run(
    tf.reduce_sum(tf.pow(pred - Y,2))/(2 * n_samples),
    feed_dict={X: np.transpose(_x), Y: _y}) #cost function as before

  print("Testing cost=", testing_cost)
  print("Absolute mean square loss difference:", abs(
        cost - testing_cost))
