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
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
#from hyperopt import hp


root = 'data/'

#############################--load_data--#####################################
temp1 = np.zeros((1400, 1))
temp2 = np.zeros((200, 1))

train = pd.read_csv(root+'train1.csv')
test = pd.read_csv(root+'test_0.5.csv')

X_train = train.iloc[:,1:-1].values
Y_train = train.iloc[:,-1].values
#Y_train = np.column_stack((Y_train, temp1))

X_test = test.iloc[:,1:-1].values
Y_test = test.iloc[:,-1].values
#Y_test = np.column_stack((Y_test, temp2))


############################--main_code--######################################


#cv_params = {'n_estimators': [400]}
#other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
#                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
#
#model = xgb.XGBRegressor(**other_params)
#optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=2, verbose=1, n_jobs=4)
#optimized_GBM.fit(X_train, Y_train)
#evalute_result = optimized_GBM.grid_scores_
#print('每轮迭代运行结果:{0}'.format(evalute_result))
#print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
#print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

model = xgb.XGBRegressor(learning_rate=0.05, n_estimators=500, max_depth=6, 
                         min_child_weight=5, seed=0, subsample=0.7, colsample_bytree=0.7,
                         gamma=0.1, reg_alpha=1, reg_lambda=1)


model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

plt.plot(Y_test)
plt.plot(Y_pred)
plt.legend(['Y_test', 'Y_pred'])
plt.show()

