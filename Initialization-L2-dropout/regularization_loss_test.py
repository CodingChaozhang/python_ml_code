# -*- coding: utf-8 -*-
"""
本程序所含内容：
1. 正则化模型：
    1.1: 不使用正则化模型 
    1.2：使用L2正则化模型
    1.3：使用随机删除节点的方法精简模型
Created on Mon Oct 29 22:59:53 2018

@author: GEAR
"""

import reg_utils
import ANN_3Layers_model as ANN #多层神经网络模型
import matplotlib.pyplot as plt


    
    
# 载入数据
train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=True)
layers_dims = [train_X.shape[0], 20, 3, 1]
plt.show()
# 训练参数
parameters = ANN.model(train_X, train_Y, layers_dims, initialize='small', 
                       num_iterations=20000, keep_prob=1, lambd=0.7)
print('训练集：')
prediction_train = ANN.prediction(train_X, train_Y, parameters)
print('测试集：')
prediction_test = ANN.prediction(test_X, test_Y, parameters)

    
