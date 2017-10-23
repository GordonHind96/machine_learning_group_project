from __future__ import division

import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier


import matplotlib.pyplot as plt
"""Read in dataset"""
set_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
column_names = ["Instance","Feature 1","Feature 2", "Feature 3","Feature 4","Feature 5","Feature 6","Feature 7",
                "Feature 8","Feature 9","Feature 10","Target","Target Class"]
dataframe = pandas.read_csv("C:\\Users\\gordo\\Desktop\\ML\\datasets\\without-noise\\The-SUM-dataset-without-noise.csv",
                             sep=';',header=0,names=column_names,index_col=0,usecols=[0,1,2,3,4,6,7,8,9,10,11],
                             nrows = 100)
data = np.array(dataframe.as_matrix())
X = data[:,:8]
y = data[:,9]
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, y)
plt.tight_layout()
plt.show()