# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 19:48:14 2018

@author: gear
github:  https://github.com/gear106
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import preprocessing
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
#from hyperopt import hp


root = 'data/'

#############################--load_data--#####################################
train = pd.read_csv(root+'train.csv')
test = pd.read_csv(root+'test.csv')

X_train = train.iloc[:,1:-1].values
Y_train = train.iloc[:,-1].values

X_test = test.iloc[:,1:-1].values
Y_test = test.iloc[:,-1].values


############################--main_code--######################################

model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=550, max_depth=4, 
                         min_child_weight=5, seed=0, subsample=0.7, colsample_bytree=0.7,
                         gamma=0.1, reg_alpha=1, reg_lambda=1)


model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

plt.plot(Y_test)
plt.plot(Y_pred)
plt.legend(['Y_test', 'Y_pred'])
plt.show()

