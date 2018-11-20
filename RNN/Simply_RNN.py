# -*- coding: utf-8 -*-
"""
简介：
实现simply-RNN

Created on Sat Nov 17 20:36:59 2018

@author: dell
"""

import numpy as np

def softmax(z):
    '''
    实现softmax函数
    '''

def rnn_cell_forward(xt, a_prev, parameters):
    '''
    实现RNN单元的单步向前传播
    Parameters:
        xt -- 时间步t输入的数据， shape=(n_x, m)
        a_prev -- 时间步t-1的隐藏层状态，shape=(n_a, m)
        parameters -- 字典：
                    Wax -- Matrix, 输入乘以权重， shape=(n_a, n_x)
                    Waa -- Matrix, 隐藏状态乘以权重， shape=(n_a, n_a)
                    Wya -- Matrix, 隐藏状态与输出相关的权重矩阵， shape=(n_y, n_a)
                    ba -- 偏置， shape=(n_a, 1)
                    by -- 偏置， 隐藏状态与输出相关的偏置， shape=(n_y, 1)
        Returns:
            a_next -- 下一个隐藏状态， shape=(n_a, m)
            yt_pred -- 在时间步t的预测， shape=(n_y, m)
            cache -- 反相传播需要的元组， 包含（a_next, a_prev, parameters)
    '''
    
    # step1 从parameters中获取参数
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    
    # step2 计算下一个激活值
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    
    # step3 计算当前单元的输出
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    
    # step3 保存反相传播需要的值
    cache = (a_next, a_prev, parameters)
    
    return a_next, yt_pred, cache

def rnn_forward(x, a0, parameters):
    '''
    实现RNN网络的向前传播
    Parameters:
        x -- 输入的全部数据， shape=(n_x, m, T_x)
        a0 -- 初始化隐藏单元， shape=(n_a, m)
        parameters -- 字典， 包含：
                    Wax -- Matrix, 输入乘以权重， shape=(n_a, n_x)
                    Waa -- Matrix, 隐藏状态乘以权重， shape=(n_a, n_a)
                    Wya -- Matrix, 隐藏状态与输出相关的权重矩阵， shape=(n_y, n_a)
                    ba -- 偏置， shape=(n_a, 1)
                    by -- 偏置， 隐藏状态与输出相关的偏置， shape=(n_y, 1)
                    
    Returns:
        a -- 所有时间步的隐藏状态， shape=(n_a, m, T_x)
        y_pred -- 所有时间步的预测， shape=(n_y, m, T_x)
        caches -- 为反相传播保存的元组， shape=(list(cache), x)
    '''
    
    # step1 初始化caches, 其包含了所有的cache
    caches = []
    
    # step2 获取x与Wya的维度信息
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape
    
    # step3 使用0来初始化a和y
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])
    
    # step4 初始化next
    a_next = a0
    
    # step5 遍历所有时间步
    for t in range(T_x):
        # 1. 使用rnn_cell_forward函数来更新next和cache
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        
        # 2.使用a来保存next
        a[:,:,t] = a_next
        
        # 3. 使用y_pred来保存预测值
        y_pred[:,:,t] = yt_pred
        
        # 4. 把cache保存到caches中
        caches.append(cache)
        
    # step6 保存反相传播需要的参数
    caches = (caches, x)
    
    return a, y_pred, caches


