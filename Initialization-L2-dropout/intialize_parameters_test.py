# -*- coding: utf-8 -*-
"""
本程序所含内容：
1. 初始化参数：
    1.1：使用0来初始化参数。
    1.2：使用随机数来初始化参数。
    1.3：使用抑梯度异常初始化参数（参见视频中的梯度消失和梯度爆炸）。

3. 梯度校验  ：对模型使用梯度校验，检测它是否在梯度下降的过程中出现误差过大的情况。

Created on Mon Oct 29 11:47:38 2018

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
parameters = ANN.model(train_X, train_Y, layers_dims, initialize='small', 
                       num_iterations=20000, keep_prob=1)
print('训练集：')
prediction_train = ANN.prediction(train_X, train_Y, parameters)
print('测试集：')
prediction_test = ANN.prediction(test_X, test_Y, parameters)
   
    

    