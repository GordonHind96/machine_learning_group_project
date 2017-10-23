print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

import pandas
from sklearn import linear_model, datasets
from sklearn.metrics import accuracy_score


print('something')

"""Read in dataset"""
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

nrows2 = set_sizes[10]

column_names = ["Instance","Feature 1","Feature 2", "Feature 3","Feature 4","Feature 5","Feature 6","Feature 7",

                "Feature 8","Feature 9","Feature 10","Target","TargetClass"]
dataframe = pandas.read_csv("C:\\Users\\bboyd\\Documents\\college - 4th year\\Machine Learning\\machine_learning_group_project\\team_13\\datasets\\sum_with_noise.csv",
                             sep=';',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,6,7,8,9,10,12],
                             nrows = nrows2)

print(nrows2)
print(int(nrows2 * .7))

X_train = dataframe.head(int(nrows2 * .7))
Y_train = X_train.TargetClass
X_train = X_train[["Feature 1","Feature 2", "Feature 3","Feature 4","Feature 6","Feature 7", "Feature 8","Feature 9","Feature 10"]]

X_test = dataframe.tail(int(nrows2 * .3))
Y_test = X_test.TargetClass
X_test = X_test[["Feature 1","Feature 2", "Feature 3","Feature 4","Feature 6","Feature 7", "Feature 8","Feature 9","Feature 10"]]

print(type(X_train))
print()


# x_train is their x , y_train is their y

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5) # was 1e5

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_train, Y_train)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].

print("check here" , type(X_train.min()))


x_min, x_max = X_train.min() - .5, X_train.max() + .5
y_min, y_max = Y_train.min() , Y_train.max()

print("aodan", Y_test.max())

#print("type of min ",type(x_min))
fl  = x_min.astype('float64', errors = 'ignore')
#print("tp", type(fl))
#print(fl)


xs_as_array = fl.as_matrix()
x_min = xs_as_array.min()
print("x_min", x_min)


xs2_as_array = x_max.as_matrix()
x_max = xs2_as_array.max()

print("xmax",x_max)

## x min & max taken care of above

print("type of min ",type(x_min))
#fl3  = x_min.astype('float64', errors = 'ignore')

print("ymin",y_min)
print("ymax",y_max)

##  metrics  ##
print(type(logreg))
y_values_predicted = logreg.predict(X_test)
print("metrics", y_values_predicted)

print("accuaracy",accuracy_score(Y_test, y_values_predicted))
