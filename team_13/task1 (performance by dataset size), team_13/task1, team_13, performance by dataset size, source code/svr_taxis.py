from __future__ import division
import time

import numpy as np
import pandas
import csv
import itertools
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.regression import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV,cross_val_score
import matplotlib.pyplot as plt


"""Read in dataset"""
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude",
                "pickup_latitude","dropoff_longitude","dropoff_latitude","store_and_fwd_flag","trip_duration"]


dataframe = pandas.read_csv("C:\\Users\\gordo\\Desktop\\ML\\datasets\\train.csv",
                             sep='\s*,\s*',header=0,names=column_names,index_col=0,usecols=[1,2,3,4,5,6,7,8,10] ,nrows =set_sizes[0])

X = dataframe["pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]
Y = dataframe["trip_duration"]
NUM_TRIALS = 30
# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}

# We will use a Support Vector Classifier with "rbf" kernel
svr = SVR(kernel="rbf")

# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

# Loop for each trial
for i in range(NUM_TRIALS):

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=svr, param_grid=p_grid, cv=inner_cv)
    clf.fit(X, Y)
    non_nested_scores[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X, y=Y, cv=outer_cv)
    nested_scores[i] = nested_score.mean()

score_difference = non_nested_scores - nested_scores

print("Average difference of {0:6f} with std. dev. of {1:6f}."
      .format(score_difference.mean(), score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
nested_line, = plt.plot(nested_scores, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
          x=.5, y=1.1, fontsize="15")

# Plot bar chart of the difference.
plt.subplot(212)
difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
plt.xlabel("Individual Trial #")
plt.legend([difference_plot],
           ["Non-Nested CV - Nested CV Score"],
           bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("score difference", fontsize="14")

plt.show()