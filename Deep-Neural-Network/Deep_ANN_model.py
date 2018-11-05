# -*- coding: utf-8 -*-
"""
简介：
建立一个具有多个隐藏层的全连接神经网络，隐藏层的激活函数全选为relu，输出层为sigimoid
参数约定：
m -------->样本个数
n_x ------>第x层的神经元数目
wxi -->维度(n_(x_1),  1) --------->第x层上第i个神经元的权重
Wx --->维度(n_x, n_(x-1)) ------->第x层的权重矩阵
bx --->维度(n_x, 1) ------------->第x层的偏置
ax --->维度(n_x, 1) ------------->一个样本第x层的激活值
Ax --->维度(n_x, m) ------------->所有样本第x层的激活值
zx --->维度(n_x, 1) ------------->一个样本第x层输入值
Zx --->维度(n_x, m) ------------->所有样本第x层输入值

-----------------------------------------------------------
Wx = [----wx1.T------
      ----wx2.T------    
           ...           
      ----wxi.T------] 此时i表示第x层的第i个神经元

Zx = [ z1         
       z2
      ... 
       zx]

z = w.T * x + b 
Ps:在我们编写神经网络代码的过程中，一个减少bug的方法就是认真检查网络中每一层的维度！

Created on Fri Oct 26 14:53:38 2018
@author: GEAR
"""
import numpy as np
import matplotlib.pyplot as plt
import data_set  # 训练和测试数据
from activation_function import *


''' --------------step1初始化W和b----------------'''
def initialize_parameters(layers_dims):
    '''
    Parameters:
        layer_dims - 储存每层神经元个数的列表
    Returns:
        parameters - 储存初始化后的W和b的字典
    '''
    np.random.seed(2)
    parameters = {}
    L = len(layers_dims)
    
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1]) / np.sqrt(layers_dims[i-1])
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
        
        # 检验数据的维度是否正确
        assert(parameters['W' + str(i)].shape == (layers_dims[i], layers_dims[i-1]))
        assert(parameters['b' + str(i)].shape == (layers_dims[i], 1))
    
    return parameters
    
def linear_forward(W, A_pre, b):
    '''
    实现向前传播的线性部分
    Parameters:
        A_pre - 上一次网络的激活值，维度为（前层节点数，样本个数）
        W - 本层的权重矩阵，维度为（本层节点数， 前层节点数）
        b - 本层的偏向量，维度为（本层节点数，1）
    Returns:
        Z - 本层的激活函数输入，维度为（本层节点数， 样本个数）
        cache - 一个包含（W, A_pre, b)的缓存，目的用来计算反向传播
    '''
    
    Z = np.dot(W, A_pre) + b
    assert(Z.shape == (W.shape[0], A_pre.shape[1]))
    
    cache = (W, A_pre, b)
    
    return Z, cache

def activation_forward(W, A_pre, b, activation=None):
    '''
    实现linear --> activation这一步骤
    Parameters:
        W - 本层的权重矩阵
        A_pre - 前层输出的激活值
        b - 本层的偏向量
        activation -激活函数类型
    Returns:
        A - 本层输出的激活函数值
        cache - 包含（W，A_pre, b) 和A的缓存
    '''
    
    
    if activation == 'sigimoid':
        Z, linear_cache = linear_forward(W, A_pre, b)
        A, activation_cache = sigimoid(Z)
    if activation == 'relu':
        Z, linear_cache = linear_forward(W, A_pre, b)
        A, activation_cache = relu(Z)
    
    assert(A.shape == Z.shape)
    cache = (linear_cache, activation_cache)
    
    return A, cache

def forward_propagation(X, parameters, layers_dims):
    '''
    计算多层网络向前传播
    Parameters:
        X - 维度（数据维数，样本个数）
        parameters - 包含各层网络权重W和偏置b的字典
        layer_dims - 包含各层网络神经元个数的列表
    Returns:
        AL - 最后一层网络的激活值
        caches - 包含各层网络权重激活值和偏向量的列表：
                 activation_forward的每个缓存值
    '''

    L = len(layers_dims) - 1 # 神经网络的层数
    caches = [] #储存各层W, A, b的缓存
    A = X #输入层的数据
    
    for i in range(1, L):
        A_pre = A  #储存前一层激活值的临时变量
        A, cache = activation_forward(parameters['W'+str(i)], A_pre, parameters['b'+str(i)],
                                      activation='relu')
        caches.append(cache)
    AL, cache = activation_forward(parameters['W'+str(L)], A, parameters['b'+str(L)],
                                      activation='sigimoid')
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))
    
    return AL, caches

def cross_entropy_loss(AL, Y):
    '''
    计算交叉熵损失函数
    Parameters:
        AL - 最后一层网络输出的激活值
        Y - 训练数据的标签
    Returns:
        loss - 交叉熵损失函数的值
    '''
    
    #检验AL和Y的维度是否相同
    assert(AL.shape == Y.shape)
    m = Y.shape[1]  #训练样本的个数
    #计算交叉熵损失函数
    probloss = np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL))
    loss = - np.sum(probloss) / m
    loss = float(np.squeeze(loss))
    assert(isinstance(loss, float))
    
    return loss

def linear_backward(dZ, cache):
    '''
    单层反向传播的线性部分
    Parmeters:
        dZ - 损失函数对当前层线性输入的梯度
        cache - 存储当前层向前传播值的元组(W, A_pre, b)
    Returns:
        dA_pre - 损失函数对前一层激活值的梯度
        dW - 损失函数对当前层权重的梯度
        db - 损失函数对当前层偏置的梯度
    '''
    W, A_pre, b = cache
    m = A_pre.shape[1]
    
    dW = np.dot(dZ, A_pre.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_pre = np.dot(W.T, dZ)
    
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_pre.shape == A_pre.shape)
    
    return dA_pre, dW, db

def activation_backward(dA, cache, activation='relu'):
    '''
    实现linear-activation层的反向传播
    Parameters:
        dA - 当前层激活值的梯度
        cache - 向前传播存储的值
        activation - 激活函数的类型
    Returns:
        dA_pre - 前一层网络激活值的梯度
        dW - 当前层的权重梯度
        db - 当前层偏置的梯度
    '''
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_pre, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigimoid':
        dZ = sigimoid_backward(dA, activation_cache)
        dA_pre, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_pre, dW, db

def backward_propagation(AL, Y, caches, layers_dims):
    '''
    多层网络的反向传播过程
    Parameters:
        AL - forward_propagation的输出值，输出层的激活值
        Y - 输入数据的标签向量
        caches - forward_propagation函数输出的缓存
    Returns:
        grads - 包含各层网络对应梯度的字典
    '''
    
    grads = {}
    L = len(layers_dims) - 1 # 网络的层数
    m = Y.shape[1] # 输入样本数量
    assert(Y.shape == AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL)) # 损失函数对输出层激活值的导数
    
    current_cache = caches[L-1]
    # 计算最后一层的梯度值
    grads['dA'+str(L)], grads['dW'+str(L)], grads['db'+str(L)] = activation_backward(dAL, current_cache, 
          activation='sigimoid')
    
    for i in reversed(range(L-1)):
        current_cache = caches[i] 
        dA_pre_i, dW_i, db_i = activation_backward(grads['dA'+str(i+2)], current_cache, activation='relu')
        grads['dA'+str(i+1)] = dA_pre_i
        grads['dW'+str(i+1)] = dW_i
        grads['db'+str(i+1)] = db_i
        
    return grads

def update_parameters(parameters, grads, learning_rate, layers_dims):
    '''
    使用梯度下降法更新参数
    Parameters:
        parameters - 包含各层权重和偏置的字典
        grads - 包含各层反向传播所需梯度的字典
        learning_rate - 学习速率
    Returns:
        parameters - 更新后的权重和偏置字典
    '''
    
    L = len(layers_dims) - 1 #神经网络的层数
    for i in range(L):
        parameters['W'+str(i+1)] -= learning_rate * grads['dW'+str(i+1)]
        parameters['b'+str(i+1)] -= learning_rate * grads['db'+str(i+1)]
        
    return parameters

def multi_layers_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000,
                       print_cost=False, figure=True):
    '''
    实现一个多层神经网络
    Parameters:
        X - 训练数据集
        Y - 训练数据标签
        learning_rate - 学习率
        num_iterations - 迭代次数
        print_cost - 是否显示损失函数值变化过程
        figure - 是否绘制误差曲线
    Returns:
        parameters - 学习到的包含各层权重矩阵和偏置向量的的字典
    '''
    
    np.random.seed(1)
    losses = []
    # step1 初始化参数
    parameters = initialize_parameters(layers_dims)
    # step2 循环梯度下降
    for i in range(num_iterations):
        # 向前传播计算每一层的激活值
        AL, caches = forward_propagation(X, parameters, layers_dims)
        # 计算损失函数
        loss = cross_entropy_loss(AL, Y)
        # 反向传播计算每一层的梯度
        grads = backward_propagation(AL, Y, caches, layers_dims)
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate, layers_dims)
        
        # 输出损失函数的值
        if i % 100 == 0:
            losses.append(loss)
            if print_cost:
                print('第 %d 次迭代，损失为%.3f' % (i, loss))
    if figure:
        plt.plot(np.squeeze(losses))
        plt.ylabel('loss')
        plt.xlabel('iterations (per tens)')
        plt.title('learning rate = %f' % learning_rate)
        plt.show()
        
    return parameters

def prediction(X, Y, parameters, layers_dims):
    '''
    该函数用于预测多层神经网络的结果
    '''
    m = X.shape[1] # 训练样本的个数
   
    AL, caches = forward_propagation(X, parameters, layers_dims)
    Y_prediction = np.zeros((1, m))
    for i in range(m):
        if AL[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    accuracy = 1 - np.mean(np.abs(Y_prediction - Y))
    print('准确率为：%.3f%%'%(100*accuracy))  
    return AL, Y_prediction

if __name__ == '__main__':
    #step1 载入训练和测试数据
    train_X, train_Y, test_X, test_Y, classes= data_set.load_dataset()
    train_X_flatten = train_X.reshape(train_X.shape[0], -1).T
    test_X_flatten = test_X.reshape(test_X.shape[0], -1).T
    
    train_X = train_X_flatten / 255
    test_X = test_X_flatten / 255
    
    # step2 开始训练网络
    layers_dims = [12288, 20, 7, 5, 1] # 网络每层神经元个数，第一个元素为输入数据维度
    parameters = multi_layers_model(train_X, train_Y, layers_dims, learning_rate=0.0075, num_iterations=500,
                       print_cost=True, figure=True)
    pred_train = prediction(train_X, train_Y, parameters, layers_dims)
    pre_test = prediction(test_X, test_Y, parameters, layers_dims)
    
    
    
    
    
        
        
        
    
        
