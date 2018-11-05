# -*- coding: utf-8 -*-
"""
简介：
建立一个只有一层隐藏层的全连接神经网络用来分类，隐藏层神经元个数为4个，
输出层神经元个数为1个，输入层个数为2个.

Created on Thu Oct 25 14:38:09 2018

@author: GEAR
"""

import numpy as np
import matplotlib.pyplot as plt
from data_set import load_planar_dataset


'''Step1--------->确定每一层神经元的个数'''

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def layer_size(X, Y):
    '''
    Parameters:
        X - 输入数据集，维度（数据维数， 数据数量）
        Y - 标签， 维度（数据维度，数据数量）
    Returns:
        n_0 - 输入层的个数
        n_1 - 第1层神经元的个数
        n_2 - 第2层神经元的个数
    '''
    n_0 = X.shape[0]  # 输入层的个数
    n_2 = Y.shape[0]  # 输出层神经元的个数
    
    return n_0, n_2

'''Step2--------->初始化每一层神经元的权重矩阵W和偏置b'''

def initialize_parameters(n_0, n_1, n_2):
    '''
    Parameters:
        n_0 - 输入层节点个数
        n_1 - 隐藏层节点个数
        n_2 - 输出层节点个数
    Returns:
        Parameters - 包含以下参数的列表：
        W1 - 第一层权重矩阵-->维度(n_1, n_0)
        b1 - 第一层偏差------>维度(n_1,  1)
        W2 - 第二层权重矩阵-->维度(n_2, n_1)
        b2 - 第二层偏差------>维度(n_2,  1)
    '''
    '''
    在神经网络中初始化权重矩阵时一般采用随机初始化矩阵，若对一个神经网络把权重初始化
    为0, 那么梯度下降将不会起作用，因为权重初始化为0，则每一层的激活单元会完全相同
    '''
    np.random.seed(2)  # 保证每次产生的随机数相同
    W1 = np.random.randn(n_1, n_0)*0.01 # 生成高斯分布,乘以0.01是为了避免权重大梯度下降慢
    b1 = np.zeros((n_1, 1))
    W2 = np.random.randn(n_2, n_1)*0.01 
    b2 = np.zeros((n_2, 1))
    
    # 确保数据格式是正确的：
    assert(W1.shape == (n_1, n_0))
    assert(b1.shape == (n_1, 1))
    assert(W2.shape == (n_2, n_1))
    assert(b2.shape == (n_2, 1))
    
    # 将初始化参数保存到字典中
    parameters = {'W1':W1,
                  'b1':b1,
                  'W2':W2,
                  'b2':b2}
    
    return parameters

'''Step3--------->向前传播，计算每一层神经元上的输入值Z和激活值A'''

def forward_propagation(X, parameters):
    '''
    Parameters:
        X - 输出数据，维度（数据维数，数据个数）
        parameters - 储存每一层神经元权重W和偏置b的字典
    Returns:
        A2 - 输出层的激活值，在反向传播时使用
        cache- 包含每一层Z和A的字典,作为反向传播的输入
    '''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    '''计算每一层的Z和A的值'''
    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)                # 隐藏层采用tanh激活函数
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)            # 输出层采用sigmoid激活函数
    
    assert(Z1.shape == (W1.shape[0], X.shape[1]))
    assert(Z2.shape == (W2.shape[0], X.shape[1]))
    
    cache = {'Z1':Z1,
             'A1':A1,
             'Z2':Z2,
             'A2':A2}
    
    return cache

'''Step3--------->计算损失函数'''

def loss_function(A2, Y):
    '''
    Parameters:
        A2 - 输出层的激活值--->维数（输出层的个数， 样本个数）
        Y  - 真实标签值------->维数（数据维度， 样本个数）
    Returns:
        loss - 交叉熵成本函数
    '''
    
    m = Y.shape[1]  # 样本个数
    logprob = np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2))
    loss = -np.sum(logprob) / m
    loss = float(np.squeeze(loss))
    
    assert(isinstance(loss, float))
    return loss

'''Step4--------->反向传播计算梯度（难点）'''

def backward_propagation(parameters, cache, X, Y) :
    '''
    Parameters:
        parameters - 储存每次神经元权重矩阵W和偏置b的字典
        cache - 储存每层神经元激活值Z和A的字典
        X - 输入数据
        Y - 标签
    Returns:
        grads - 包含每层损失函数对神经元权重和偏置偏导数的字典
    '''
    m = X.shape[1]   # 输入样本个数
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    '''反向传播过程（难点）'''
    dZ2 = A2 - Y   
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))  # 这里比较难理解
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {'dW1':dW1,
             'db1':db1,
             'dW2':dW2,
             'db2':db2}
    
    return grads

'''Step5--------->采用梯度下降法更新权重W和偏差b'''

def update_parameters(parameters, grads, learning_rate=0.01):
    '''
    Parameters:
        parameters - 包含各层神经元权重矩阵和偏置的字典
        grads - 包含损失函数对各层神经元权重和偏置的导数
        learning_rate - 学习速率
    Returns:
        parameters - 更新后的各层神经元权重矩阵和偏置
    '''
    
    W1, W2 = parameters['W1'], parameters['W2']
    b1, b2 = parameters['b1'], parameters['b2']
    
    dW1, dW2 = grads['dW1'], grads['dW2']
    db1, db2 = grads['db1'], grads['db2']
    
    '''梯度下降更新参数'''
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {'W1':W1,
                  'b1':b1,
                  'W2':W2,
                  'b2':b2}
    
    return parameters


'''Step5--------->将上述函数整合到一个model函数中'''   

def ANN_model(X, Y, n_1, num_iterations, learning_rate=0.5, print_loss=False):
    '''
    Parameters:
        X - 输入数据集
        Y - 输入标签
        n_1 - 第一层神经元的个数
        num_iterations - 梯度下降法迭代的次数
        print_loss - 结果中是否输出损失函数变化情况
    Returns:
        parameters - 模型最后学习到的权重和偏置参数
    '''
    n_0, n_2 = layer_size(X, Y)
    
    '''初始化每层的权重矩阵和偏置'''
    parameters = initialize_parameters(n_0, n_1, n_2)
    W1, W2 = parameters['W1'], parameters['W2']
    b1, b2 = parameters['b1'], parameters['b2']
    
    '''梯度下降更新parameters'''
    for i in range(num_iterations):
        # step1 计算各层激活值
        cache = forward_propagation(X, parameters)
        # step2 计算损失函数
        loss = loss_function(cache['A2'], Y)
        # step3 反向传播计算各层梯度
        grads = backward_propagation(parameters, cache, X, Y)
        # step4 梯度下降更新权重参数
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_loss:
            if i % 100 == 0: # 每迭代100次输出损失函数的值
                print('第 %d 次迭代，loss = %f' % (i, loss))
    return parameters


'''Step6--------->对输入的测试集进行预测'''
def prediction(parameters, X):
    '''
    Parameters:
        parameters - 神经网络训练后更新的参数值
        X - 输入新的测试集数据，维度（数据维数，样本个数）
    Returns:
        predictions - 神经网络预测的结果值
    '''
    # step1 向前传播计算测试集输出的结果
    cache = forward_propagation(X, parameters)
    A2 = cache['A2'] # 输出层输出的结果
    # 这里做的是二分类预测，输出值大于0.5设置为1，否则为0
    predicions = np.round(A2)
    
    return predicions.astype(np.int8)


if __name__ == '__main__':
    X, Y = load_planar_dataset()
    # step1训练数据网络
    parameters = ANN_model(X, Y, n_1=50, num_iterations=10000, learning_rate=0.8, print_loss=True)
    Y_prediction = prediction(parameters, X)
    train_accuracy = 1 - np.mean(np.abs(Y_prediction - Y))
    print('训练准确率为：%.3f%%'%(100*train_accuracy))
    
    
    


    


    

    
        
        
        
    
    