# 245,057 rows

from mlxtend.plotting import plot_decision_regions
from mlxtend.classifier import LogisticRegression
from sklearn.model_selection import cross_val_predict as cvsc
import pandas as pd
from sklearn import linear_model


files = "C:\\Users\\bboyd\\Documents\\college - 4th year\\Machine Learning\\machine_learning_group_project\\team_13\\datasets\\skin.csv"
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Feature 1","Feature 2", "Feature 3","Target"]

nrows = 6
data_size = set_sizes[nrows]
dataframe = pd.read_csv(files,
                             sep=',',header=0,names=column_names,usecols=[0,1,2,3],
                             nrows = data_size)


dataframe.dropna(axis=0, how='all')
X = dataframe[["Feature 1","Feature 2","Feature 3"]]
Y = dataframe ["Target"]

X = X.get_values()

print(type(Y))
Y = Y.get_values()

X_train = X[:int(set_sizes[nrows]*0.7) ]
X_test = X[int(set_sizes[nrows]*0.7) : ]
print("X" , X.size)
print("x_train", X_train.size)
print("xtest", X_test.size)

len = Y.size;
train_len = (int) ((len*.7))
Y_train = Y[ : train_len ]

test_len = (int) (len - train_len)

Y_test = Y[ train_len : ]

print("Y" , Y.size)
print("y_train", Y_train.size)
print("y test", Y_test.size)

print (Y_test.size)
print (X_test.size)

lr = LogisticRegression(eta=0.1,
                        l2_lambda=0.0,
                        epochs=100,
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
print("ACC 70-30 split" , acc)


logreg = linear_model.LogisticRegression(C=1e5)

logreg.fit(X_train, Y_train)

cv = (cvsc( logreg, X, Y, cv=10))


correct = 0
total = 0

i2 = 0
while(i2 < Y_test.size):
    if(Y_test[i2] == cv[i2]):
        correct += 1
    i2+=1
    total += 1

print("--------")
print(pre[0])
print(Y_test[0])

acc = correct/total
print("cv split" , acc)


cv = (cvsc( logreg, X, Y, cv=10))
correct = 0
total = 0

i2 = 0
while(i2 < Y_test.size):
    if(Y_test[i2] == cv[i2]):
        correct += 1
    i2+=1
    total += 1

print("--------")
print(pre[0])
print(Y_test[0])

acc = correct/total
print("cv split" , acc)



print(cvsc( logreg, X, Y, cv=10))
