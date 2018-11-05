# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 20:52:23 2018

@author: GEAR
"""

import init_utils
import ANN_3Layers_model as ANN #多层神经网络模型
import matplotlib.pyplot as plt

# 载入数据
train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=True)
layers_dims = [train_X.shape[0], 10, 5, 1]
plt.show()

# 训练参数
# 1.常规梯度下降
#parameters = ANN.model(train_X, train_Y, layers_dims, initialize='small', num_iterations=20000, keep_prob=1)
## 2.随机梯度下降
#parameters = ANN.model_SGD(train_X, train_Y, layers_dims, num_iterations=1000,initialize='small')
# 3.mini_batch梯度下降
parameters = ANN.model_mini_batch(train_X, train_Y, layers_dims, mini_batch_size=100, num_iterations=20000,initialize='small')
print('训练集：')
prediction_train = ANN.prediction(train_X, train_Y, parameters)
print('测试集：')
prediction_test = ANN.prediction(test_X, test_Y, parameters)