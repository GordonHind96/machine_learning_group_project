#Support Vector Machine
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import ops
ops.reset_default_graph()
#create graph
sesh = tf.Session()

set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Feature 1","Feature 2", "Feature 3","Target"]

i = set_sizes[8]
dataframe = pd.read_csv("C:\\Users\\gordo\\Desktop\\ML\\machine_learning_group_project\\team_13\\datasets\\skin.csv",
                             sep=',',header=0,names=column_names,usecols=[0,1,2,3],
                             nrows =i)
                             	#nrows = set_sizes[4]
dataframe = dataframe.sample(frac=1)
X_train = dataframe.head(int(i * .7)) 
#X_train.append(dataframe.tail(int(nrows2*.35)))
Y_train = X_train.Target
X_train = X_train[["Feature 1","Feature 2","Feature 3"]]
X_test = dataframe.head(int(i * .3))
#X_test.append(dataframe.tail(int(nrows2 * .15)))

Y_test = X_test.Target
X_test = X_test[["Feature 1", "Feature 2","Feature 3"]]

#Declare batch size
batch_size = 100

# Initilaize placeholder
x_data =  tf.placeholder("float")
y_target = tf.placeholder("float")

#Create variables for linear regrression
A = tf.Variable(tf.random_normal(shape=[3,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

#Declare model operations
model_output = tf.subtract(tf.matmul(x_data,A),b)

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
for i in range(500):
	rand_index = np.random.choice(len(X_train),size=batch_size)
	rand_x = X_train.iloc(rand_index)
	rand_y = np.transpose([Y_train.iloc(rand_index)])
	sesh.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})

	temp_loss =sesh.run(loss,feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(temp_loss)
	train_acc_temp = sesh.run(accuracy, feed_dict={
		x_data: X_train,
		y_target: np.transpose([Y_train])})
	train_accuracy.append(train_acc_temp)

	test_acc_temp = sesh.run(accuracy,feed_dict={
		x_data: X_test,
		y_target: np.transpose([Y_test])})
	test_accuarcy.append(test_acc_temp)

	if(i+1) % 100 == 0:
		print('Step #{} A = {}, b = {}'.format(
			str(i+1),
			str(sess.run(A)),
			str(sess.run(b))
			))
		print('Loss = ' + str(temp_loss))

			


