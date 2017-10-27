#Support Vector Machine
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import ops
ops.reset_default_graph()
#create graph
sesh = tf.Session()


column_names = ["Feature 1","Feature 2", "Feature 3","Target"]
n_samples = 100000

dataframe = pd.read_csv("C:\\Users\\gordo\\Desktop\\ML\\machine_learning_group_project\\team_13\\datasets\\skin.csv",
                             sep=',',header=0,names=column_names,usecols=[0,1,2,3],
                             nrows =n_samples)
                             	#nrows = set_sizes[4]
dataframe = dataframe.sample(frac=1)
Y = dataframe["Target"]
X = dataframe[["Feature 1","Feature 2","Feature 3"]]
X = X.get_values()
#print(X)

X_train = X[:int(n_samples*0.7)]
X_test = X[int(n_samples*0.7):]

Y_train = Y[:int(n_samples*0.7)]
Y_test = Y[int(n_samples*0.7):]
print(Y_train)
#Declare batch size
batch_size = 100

# Initilaize placeholder
x_data =  tf.placeholder("float")
y_target = tf.placeholder("float")

#Create variables for linear regrression
A = tf.Variable(tf.random_normal(shape=[3,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

#Declare model operations
model_output = tf.subtract(tf.multiply(x_data,A),b)

#Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(A))

#Declare loss function 
#Loss = max(0,1-pred*actaul) + alpha * L2_norm(A)^2
#L2 regularization parameter, alpha
alpha = tf.constant([0.01])
#Margin term in loss
classification_term = tf.reduce_mean(tf.maximum(0.,tf.subtract(1.,tf.multiply(model_output,y_target))))
#put terms together
loss = tf.add(classification_term,tf.multiply(alpha,l2_norm))

#Declare prediciton function
prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,y_target),tf.float32))

#declare optimizer 
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

#Initialize variables
init = tf.global_variables_initializer()
sesh.run(init)

#Training loop
loss_vec = []
train_accuracy = []
test_accuarcy = []
for c in range(100):
	print(c)
	for (x,y) in zip(X_train, Y_train):
		sesh.run(train_step, feed_dict={x_data:x , y_target:y})
		temp_loss =sesh.run(loss,feed_dict={x_data: x, y_target: y})
	loss_vec.append(temp_loss)
	train_acc_temp = sesh.run(accuracy, feed_dict={
		x_data: np.transpose(X_train),
		y_target: Y_train})
	train_accuracy.append(train_acc_temp)

	test_acc_temp = sesh.run(accuracy,feed_dict={
		x_data: np.transpose(X_test),
		y_target: Y_test})
	test_accuarcy.append(test_acc_temp)

	if(c+1) % 100 == 0:
		print('Step #{} A = {}, b = {}'.format(
			str(c+1),
			str(sess.run(A)),
			str(sess.run(b))
			))
		print('Loss = ' + str(temp_loss))

			


