# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:20:47 2018

@author: GEAR
"""

import init_utils
import ANN_3Layers_model as ANN #多层神经网络模型
import matplotlib.pyplot as plt

# 载入数据
train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=True)
layers_dims = [train_X.shape[0], 10, 5, 1]
plt.show()


## mini_batch梯度下降
#parameters = ANN.model_mini_batch(train_X, train_Y, layers_dims, mini_batch_size=100, num_iterations=20000,initialize='small')
# momentum梯度下降
parameters = ANN.model_opt(train_X, train_Y, layers_dims, optimizer='gd', initialize='random',learning_rate=0.003, mini_batch_size=128, num_epochs=10000, epsilon=1e-5)
print('训练集：')
prediction_train = ANN.prediction(train_X, train_Y, parameters)
print('测试集：')
prediction_test = ANN.prediction(test_X, test_Y, parameters)