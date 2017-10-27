import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

rng = np.random
#params
learning_rate = 0.01
training_epochs = 10
display_steps = 50

set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

column_names = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude"
        ,"dropoff_longitude","dropoff_latitude","store_and_fwd_flag","trip_duration"]
"""Read in dataset"""
i=6
dataframe = pd.read_csv(tf.gfile.Open("C:\\Users\\gordo\\Desktop\\ML\\datasets\\train.csv"),
sep=',',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,5,6,7,8,10] ,nrows =set_sizes[i])

Y = dataframe["trip_duration"]
X = dataframe[["passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]]
X = X.get_values()
#print(X)
X_train = X[:int(set_sizes[i]*0.7)]
X_test = X[int(set_sizes[i]*0.7):]

Y_train = Y[:int(set_sizes[i]*0.7)]
Y_test = Y[int(set_sizes[i]*0.7):]


# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# model weights set to random
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

#Linear model construction
pred = tf.add(tf.multiply(X,W),b)

#Mean Squared Error
cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*set_sizes[i])

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
    for (x,y) in zip(X_train, Y_train):
      sesh.run(alpha, feed_dict={X: x, Y: y})

      #logs
      if(epoch+1) % display_steps == 0:
         c = sesh.run(cost, feed_dict={X:X_train,Y:Y_train})
         print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),\
          "W=",sess.run(W),"b=", sess.run(b))

  print("FInished")

  plt.plot(X_train,Y_train,'ro')
  plt.plot(X_train,sesh.run(W)* X_train + sesh.run(b))
  plt.show()

  #Test
  print("Testing Mean Squared Error")
  #print("X_test shape:",X_test.shape())
  testing_cost = sesh.run(
    tf.reduce_sum(tf.pow(pred - Y,2))/(2 * set_sizes[i]),
    feed_dict={X: np.transpose(X_test), Y: Y_test}) #cost function as before

  print("Testing cost=", testing_cost)
  print("Absolute mean square loss difference:", abs(
        cost - testing_cost))
  plt.plot(X_test, Y_test,'bo')
  plt.plot(X_train, sesh.run(W) * X_train + sesh.run(b))
  plt.show()