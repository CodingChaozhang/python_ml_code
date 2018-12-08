# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:40:14 2018

@author: GEAR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadData(root):
    '''
    Parameters:
        root - 输入数据文件路径
    Returns:
        X - 处理后的输入数据，维度（数据维数，样本个数）
        Y - 处理后的样本标签，维度（数据维度，样本个数）
    '''
    df = pd.read_table(root, header=None)
    # X(nx,m),其中nx表示一个样本的维数，m表示样本数   
    X = df.iloc[:,0:2].values.T
    # Y(1, m),其中m表示样本数
    Y = df.iloc[:,-1].values.reshape(1, df.shape[0]) #这里出来的shape为(100,),为了防止出现意外bug
    
    # 绘图
    X0 = df[df.iloc[:,2]==0] # 类别为0的数据
    X1 = df[df.iloc[:,2]==1] # 类别为1的数据
    plt.scatter(X0.iloc[:,0], X0.iloc[:,1], s=20, c='green', marker='s', alpha=.5)
    plt.scatter(X1.iloc[:,0], X1.iloc[:,1], s=20, c='red', alpha=.5)
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-8.24 - 0.57*x) / -1.32
    plt.plot(x, y)
    plt.title('DataSet')
    
    return X, Y

def sigimod(z):
    '''定义sigimod函数'''
    return 1 / (1 + np.exp(-z))

def initialize(dim):
    '''
    定义参数初始化函数
    Parameters:
        dim - 输入样本数据的维数
    Returns:
        W - 权重矩阵
        b - 偏差
    '''
    W = np.zeros((dim, 1))
    b = 0
    
    return W, b

def propagate(W, b, X, Y, SGD):
    '''
    定义传播函数
    Parameters:
        W - 权重矩阵------>维度（数据维数， 1）
        b - 偏置---------->常数
        X - 输入数据------>维度（数据维数，样品个数）
        Y - 输入标签------>维度（1， 样品个数）
        SGD - 是否采用随机梯度下降法
    Returns:
        dW - 损失函数对权重矩阵的偏导数
        db - 损失函数对偏置的偏导数
        loss - 损失函数
    '''
    m = X.shape[1]  # 样本个数
    n = X.shape[0]  # 数据维数
    
    # step1------------------------------>正向传播计算激活值
    Z = np.dot(W.T, X) + b
    A = sigimod(Z)
    
    # step2------------------------------>正向传播计算损失函数
    logprob = np.multiply(Y, np.log(A)) + np.multiply(1-Y, np.log(1-A))
    loss = - np.sum(logprob) / m
    
    # step3------------------------------>反向传播计算梯度值
    dZ = A - Y
    index = int(np.random.randint(0, m))# 随机选取一个样本序列号
    if SGD:
        dW = np.dot(X[:,index].reshape(n, 1), float(dZ[:,index]))
        db = float(dZ[:,index])
    else:
        dW = (1/m) * np.dot(X, dZ.T)
        db = (1/m) * np.sum(A - Y)
    assert(dW.shape == W.shape)
    loss = np.squeeze(loss)
    
    return dW, db, loss

def optimize(W, b, X, Y, num_iterations, learning_rate, SGD, print_loss=False):
    '''
    采用梯度下降法对参数进行更新
    Parameters:
        W - 初始化的权重矩阵------>维度（数据维数， 1）
        b - 初始化的偏置---------->常数
        num_iterations - 梯度下降迭代次数
        learning_rate - 学习速率
        SGD - 是否采用随机梯度下降法
        print_loss - 是否输出损失函数变化情况
    Returns:
        W - 更新后的权重矩阵
        b - 更新后的偏置
        loss - 训练后的损失函数的值
    '''
    
    # 进行梯度下降    
    for i in range(num_iterations):
        dW, db, loss = propagate(W, b, X, Y, SGD)
        W = W - learning_rate*dW
        b = b - learning_rate*db
        
        if print_loss and i % 100 == 0:
            print('loss after iteration %i: %f' % (i, loss))
        
    return W, b, loss


def predict(W, b, X):
    '''
    对训练后输出的值进行预测
    Parameters:
        W - 更新后的权重矩阵
        b - 更新后的偏置
        X - 输入的训练数据
    Returns:
        Y_prediction - 预测的标签值
    '''
    
    m = X.shape[1]  # 样本个数
    A = sigimod(np.dot(W.T, X) + b)
    Y_prediction = np.round(A)
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def trainModel(X_train, Y_train, num_iterations=5000, learning_rate=0.005, SGD=True, print_loss=False):
    
    dim = X_train.shape[0]
    # step1 初始化训练参数
    W, b = initialize(dim)
    # step2 计算更新后的参数
    W, b, losss = optimize(W, b, X_train, Y_train, num_iterations, learning_rate, SGD, print_loss)
    # step3 对输出的值进行预测
    Y_prediction_train = predict(W, b, X_train)
    # step4 计算模型准确率
    train_accuracy = 1 - np.mean(np.abs(Y_prediction_train - Y_train))
    print('训练准确率为：%.3f%%'%(100*train_accuracy))
    
    return W, b, Y_prediction_train, train_accuracy
 
      
if __name__ == '__main__':
    #数据集地址
    root = 'data/testSet.txt'
    X_train, Y_train = loadData(root)
    W, b, Y_prediction, accuracy = trainModel(X_train, Y_train, num_iterations=5000, learning_rate=0.05, SGD=True, print_loss=True)
    
    
    
