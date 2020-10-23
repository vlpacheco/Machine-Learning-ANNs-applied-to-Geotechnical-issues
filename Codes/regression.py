# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:54:48 2020

@author: Luci
"""

import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

dataset = np.loadtxt("mico_doe.csv", delimiter=",")


X = dataset[:,0:3]
Y = np.expand_dims(dataset[:,3],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size = 0.15,
                                                    random_state = 10)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train) 

predicted_regr = regr.predict(X_test)

predicted_regr2 = regr.predict(X)

# plt.plot(np.arange(len(Y)),real,'o',np.arange(len(Y)),
#          predictions,'o',np.arange(len(Y)),predicted_regr,'o')

# plt.legend(['Real', 'RNA','Regr'], loc='upper left')
