# -*- coding: utf-8 -*-
"""
简介：
建立一个只有两个隐藏层的全连接网络：
1. 测试3中不同的参数初始化情况对结果的影响
2. 测试加入L2正则化和非正则化的区别
3. 测试加入Dropout的区别

Created on Tue Oct 30 21:34:12 2018

@author: GEAR
"""
import math
import numpy as np
import activation_function as AF
import matplotlib.pyplot as plt
from sys import exit

def initialize_parameters_zeros(layers_dims):
    '''
    采用0初始化权重矩阵
    '''
    parameters = {}
    
    L = len(layers_dims)
    
    for i in range(1, L):
        parameters['W'+str(i)] = np.zeros((layers_dims[i], layers_dims[i-1]))
        parameters['b'+str(i)] = np.zeros((layers_dims[i], 1))
        
    return parameters

def initialize_parameters_random(layers_dims):
    '''
    采用较大的数值初始化权重矩阵
    '''
    parameters = {}
    
    L = len(layers_dims)
    
    for i in range(1, L):
        parameters['W'+str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1])*2
        parameters['b'+str(i)] = np.zeros((layers_dims[i], 1))
        
    return parameters

def initialize_parameters_sqrt(layers_dims):
    '''
    采用较大的数值初始化权重矩阵
    '''
    parameters = {}
    
    L = len(layers_dims)
    
    for i in range(1, L):
        parameters['W'+str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1])*np.sqrt(2/layers_dims[i-1])
        parameters['b'+str(i)] = np.zeros((layers_dims[i], 1))
        
    return parameters

def forward_propagation(X, parameters):
    '''
    '''
    # 初始化参数的值
    W1 = parameters['W1']
    b1 = parameters['b1']
    
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    # 正向传播
    Z1 = np.dot(W1, X) + b1
    A1 = AF.relu(Z1)
    
    Z2 = np.dot(W2, A1) + b2
    A2 = AF.relu(Z2)
    
    Z3 = np.dot(W3, A2) + b3
    A3 = AF.sigimoid(Z3)
    
    cache = (A1, Z1, W1, b1, A2, Z2, W2, b2, A3, Z3, W3, b3)
    
    return A3, cache

def forward_propagation_dropout(X, parameters, keep_prob=0.5):
    '''
    加入了dropout的向前传播,输出层不加dropout
    '''
    # 初始化参数的值
    np.random.seed(1)
    W1 = parameters['W1']
    b1 = parameters['b1']
    
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    # 正向传播
    Z1 = np.dot(W1, X) + b1
    A1 = AF.relu(Z1)
    # 加入dropout
    D1 = np.random.rand(A1.shape[0], A1.shape[1])  #step1 初始换一个和A1相同维数的矩阵
    D1 = D1 < keep_prob                            #step2 将D1的值转换为0或1
    A1 = A1 * D1                                   #step3 舍弃A1中的一些节点, 将其激活值变为0
    A1 = A1 / keep_prob                            #step4 缩放未舍弃的节点的值
    
    Z2 = np.dot(W2, A1) + b2
    A2 = AF.relu(Z2)
    # 加入dropout
    D2 = np.random.rand(A2.shape[0], A2.shape[1])  #step1 初始换一个和A2相同维数的矩阵
    D2 = D2 < keep_prob                            #step2 将D2的值转换为0或1
    A2 = A2 * D2                                   #step3 舍弃A2中的一些节点, 将其激活值变为0
    A2 = A2 / keep_prob                            #step4 缩放未舍弃的节点的值
    
    Z3 = np.dot(W3, A2) + b3
    A3 = AF.sigimoid(Z3)
    
    cache = (A1, D1, Z1, W1, b1, A2, D2, Z2, W2, b2, A3, Z3, W3, b3)
    
    return A3, cache

def cross_entropy_loss(A3, Y):
    '''
    计算交叉熵损失函数
    '''
    m = Y.shape[1]
    logprob = np.multiply(Y, np.log(A3)) + np.multiply(1-Y, np.log(1-A3))
    loss = - np.sum(logprob) / m
    
    return loss

def cross_entropy_loss_L2(A3, Y, parameters, lambd):
    '''
    L2正则化损失函数
    '''
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    
    cross_entropy = cross_entropy_loss(A3, Y)
    L2_reg = lambd *(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2*m)
    
    loss = cross_entropy + L2_reg
    
    return loss


def backward_propagation(X, Y, cache):
    '''
    反向传播
    '''
    (A1, Z1, W1, b1, A2, Z2, W2, b2, A3, Z3, W3, b3) = cache
    m = Y.shape[1] # 样本个数
    
    dA3 = - (np.divide(Y, A3) - np.divide(1-Y, 1-A3))
    dZ3 = AF.sigimoid_backward(dA3, Z3)
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = AF.relu_backward(dA2, Z2)
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = AF.relu_backward(dA1, Z1)
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {'dW1':dW1, 'dW2':dW2, 'dW3':dW3,
             'db1':db1, 'db2':db2, 'db3':db3}
    
    return grads

def backward_propagation_L2(X, Y, cache, lambd):
    '''
    反向传播,只在计算dW时发生了变化，其他地方不变
    '''
    (A1, Z1, W1, b1, A2, Z2, W2, b2, A3, Z3, W3, b3) = cache
    m = Y.shape[1] # 样本个数
    
    
    dA3 = - (np.divide(Y, A3) - np.divide(1-Y, 1-A3))
    dZ3 = AF.sigimoid_backward(dA3, Z3)
    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd * W3 / m)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = AF.relu_backward(dA2, Z2)
    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd * W2 / m)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = AF.relu_backward(dA1, Z1)
    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd * W1 / m)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {'dW1':dW1, 'dW2':dW2, 'dW3':dW3,
             'db1':db1, 'db2':db2, 'db3':db3}
    
    return grads

def backward_propagation_dropout(X, Y, cache, keep_prob):
    '''
    加入dropout的反向传播
    '''
    (A1, D1, Z1, W1, b1, A2, D2, Z2, W2, b2, A3, Z3, W3, b3) = cache
    m = Y.shape[1] # 样本个数
    
    dA3 = - (np.divide(Y, A3) - np.divide(1-Y, 1-A3))
    dZ3 = AF.sigimoid_backward(dA3, Z3)
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2                      # step1 使用正向传播时未关闭的节点
    dA2 = dA2 / keep_prob               # step2 缩放未舍弃节点的值
    
    dZ2 = AF.relu_backward(dA2, Z2)
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1                      # step1 使用正向传播时未关闭的节点
    dA1 = dA1 / keep_prob               # step2 缩放未舍弃节点的值
    
    dZ1 = AF.relu_backward(dA1, Z1)
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {'dW1':dW1, 'dW2':dW2, 'dW3':dW3,
             'db1':db1, 'db2':db2, 'db3':db3}
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    '''
    使用梯度下降法更新参数
    Parameters:
        parameters - 包含各层权重和偏置的字典
        grads - 包含各层反向传播所需梯度的字典
    Returns:
        parameters - 更新后的权重和偏置字典
    '''
    
    L = len(parameters) // 2
    
    for i in range(L):
        parameters['W'+str(i+1)] -= learning_rate * grads['dW'+str(i+1)]
        parameters['b'+str(i+1)] -= learning_rate * grads['db'+str(i+1)]
    
    return parameters

def initilize_V(parameters):
    '''
    初始化momentum梯度下降法的中间变量VdW,Wdb
    '''
    L = len(parameters) // 2
    V = {}
    for i in range(L):
        V['dW'+str(i+1)] = np.zeros_like(parameters['W'+str(i+1)])
        V['db'+str(i+1)] = np.zeros_like(parameters['b'+str(i+1)])
        
    return  V

def initilize_S(parameters):
    '''
    初始化RMSProp梯度下降法的中间变量SdW,Sdb
    '''
    L = len(parameters) // 2
    S = {}
    for i in range(L):
        S['dW'+str(i+1)] = np.zeros_like(parameters['W'+str(i+1)])
        S['db'+str(i+1)] = np.zeros_like(parameters['b'+str(i+1)])
        
    return  S

def update_parameters_momentum(parameters, grads, V, beta1, learning_rate):
    '''
    采用momentum梯度下降法更新参数
    Parameters:
        V - 包含VdW和Vdb的字典
        beta1 - 超参数
    Returns:
        parameters - 更新后的参数字典
    '''
    L = len(parameters) // 2
    
    for i in range(L):
        # step1 计算VdW, Vdb
        V['dW'+str(i+1)] = beta1 * V['dW'+str(i+1)] + (1 - beta1)*grads['dW'+str(i+1)]
        V['db'+str(i+1)] = beta1 * V['db'+str(i+1)] + (1 - beta1)*grads['db'+str(i+1)]
        
        # step2 更新参数
        parameters['W'+str(i+1)] -= learning_rate * V['dW'+str(i+1)]
        parameters['b'+str(i+1)] -= learning_rate * V['db'+str(i+1)]
        
    return parameters, V


def update_parameters_RMSProp(parameters, grads, S, beta2, learning_rate):
    '''
    采用RMSProp(Root mean square prop)梯度下降法更新参数
    Parameters:
        S - 包含SdW和Sdb的字典
        beta2 - 超参数
    Returns:
        parameters - 更新后的参数字典
    '''
    L = len(parameters) // 2
    
    for i in range(L):
        # step1 计算SdW, Sdb
        S['dW'+str(i+1)] = beta2 *S['dW'+str(i+1)] + (1 - beta2)*np.square(grads['dW'+str(i+1)])
        S['db'+str(i+1)] = beta2 *S['db'+str(i+1)] + (1 - beta2)*np.square(grads['db'+str(i+1)])
        
        # step2 更新参数
        parameters['W'+str(i+1)] -= learning_rate * grads['dW'+str(i+1)] / (np.sqrt(S['dW'+str(i+1)]) + 1e-8)
        parameters['b'+str(i+1)] -= learning_rate * grads['db'+str(i+1)] / (np.sqrt(S['db'+str(i+1)])  + 1e-8)  
        
    return parameters, S

def update_parameters_Adam(parameters, grads, V, S, beta1, beta2, t, learning_rate, epsilon=1e-8):
    '''
    采用Adam梯度下降法更新参数
    '''
    L = len(parameters) // 2
    V_corrected = {}
    S_corrected = {}
    
    for i in range(L):
        # step1 计算Vdw, Vdb
        V['dW'+str(i+1)] = beta1 * V['dW'+str(i+1)] + (1 - beta1)*grads['dW'+str(i+1)]
        V['db'+str(i+1)] = beta1 * V['db'+str(i+1)] + (1 - beta1)*grads['db'+str(i+1)]
        
        # step2 修正VdW, Vdb
        V_corrected['dW'+str(i+1)] = V['dW'+str(i+1)] / (1 - beta1**t)
        V_corrected['db'+str(i+1)] = V['db'+str(i+1)] / (1 - beta1**t)
        
        # step3 计算Sdw, Sdb
        S['dW'+str(i+1)] = beta2 *S['dW'+str(i+1)] + (1 - beta2)*np.square(grads['dW'+str(i+1)])
        S['db'+str(i+1)] = beta2 *S['db'+str(i+1)] + (1 - beta2)*np.square(grads['db'+str(i+1)])
    
        # step4 修正SdW, Sdb
        S_corrected['dW'+str(i+1)] = S['dW'+str(i+1)] / (1 - np.power(beta2, t))
        S_corrected['db'+str(i+1)] = S['db'+str(i+1)] / (1 - np.power(beta2, t))
        
        # step5 更新parameters
        parameters['W'+str(i+1)] -= learning_rate * V_corrected['dW'+str(i+1)] / (np.sqrt(S_corrected['dW'+str(i+1)]) + epsilon)
        parameters['b'+str(i+1)] -= learning_rate * V_corrected['db'+str(i+1)] / (np.sqrt(S_corrected['db'+str(i+1)]) + epsilon)
        
        return parameters, V, S
    

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    '''
    将训练数据分为多个mini_batch进行训练
    '''
    np.random.seed(seed)
    m = Y.shape[1]
    mini_batchs = []
    
    # step1 打乱数据集
    permutation = list(np.random.permutation(m)) # 返回一个长度为m的随机数组，里边数字为0到m-1
    shuffled_X = X[:,permutation] # 将每一列的数据按照permutaion来排列
    shuffled_Y = Y[:,permutation].reshape(1, m)
    
    # step2 分割数据集
    mini_batch_num = math.floor(m / mini_batch_size)
    for k in range(mini_batch_num):
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batchs.append(mini_batch)
    
    # step3 处理剩余数据(若m不是mini_batch_size的整数倍，则有数据剩余)
    if m % mini_batch_size != 0:
        # 获取剩余的数据
        mini_batch_X = shuffled_X[:, mini_batch_size*mini_batch_num:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size*mini_batch_num:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batchs.append(mini_batch)
    
    return mini_batchs


def model(X, Y, layers_dims, learning_rate=0.01, num_iterations=15000, print_cost=True, 
          initialize='random', figure=True, lambd=0, keep_prob=1):
    '''
    实现一个3层网络模型
    '''
    costs = []
    
    # step1 初始化参数
    if initialize == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialize == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialize == 'small':
        parameters = initialize_parameters_sqrt(layers_dims)
    else:
        print('初始化参数出错! 程序结束')
        exit
    
    # step2 开始学习
    for i in range(num_iterations):
        # 向前传播
        if keep_prob == 1:  #不采用dropout
            A3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1: #采用dropout
            A3, cache = forward_propagation_dropout(X, parameters, keep_prob)
        else:
            print('dropout 参数出错! 程序结束')
            exit
        
        # 计算损失函数
        if lambd == 0:  # 不采用L2正则化
            loss = cross_entropy_loss(A3, Y)
        else:           # 采用L2正则化
            loss = cross_entropy_loss_L2(A3, Y, parameters, lambd)
        
        # 反向传播
        if lambd == 0 and keep_prob == 1:              # 不采用L2，不采用dropout
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0 and keep_prob == 1:            # 采用L2，不采用dropout
            grads = backward_propagation_L2(X, Y, cache, lambd)
        elif lambd == 0 and keep_prob < 1 :            # 采用dropout，不采用L2
            grads = backward_propagation_dropout(X, Y, cache, keep_prob)
        else:
            print('采用dropout+L2的子程序未定义')
            exit 
        
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 1000 == 0:
            costs.append(loss)
            if print_cost:
                print('第 %d 次迭代，损失为%.3f' % (i, loss))
    if figure:
        plt.plot(np.squeeze(costs))
        plt.ylabel('loss')
        plt.xlabel('iterations (per tens)')
        plt.title('learning rate = %f' % learning_rate)
        plt.show()
        
    return parameters

def model_SGD(X, Y, layers_dims, learning_rate=0.01, num_iterations=1500, print_cost=True, 
          initialize='random', figure=True):
    '''
    实现一个3层网络模型,采用随机梯度下降法
    '''
    costs = []
    
    # step1 初始化参数
    if initialize == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialize == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialize == 'small':
        parameters = initialize_parameters_sqrt(layers_dims)
    else:
        print('初始化参数出错! 程序结束')
        exit
    
    m = Y.shape[1]
    # step2 开始学习
    for i in range(num_iterations):
        for j in range(m):
            # 向前传播
            A3, cache = forward_propagation(X[:,j].reshape(-1, 1), parameters)    
            # 计算损失
            loss = cross_entropy_loss(A3, Y[:,j].reshape(-1,1))
            # 反向传播
            grads = backward_propagation(X[:,j].reshape(-1,1), Y[:,j].reshape(-1,1), cache)       
            # 更新参数
            parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 100 == 0:
            costs.append(loss)
            if print_cost:
                print('第 %d 次迭代，损失为%.3f' % (i, loss))
    if figure:
        plt.plot(np.squeeze(costs))
        plt.ylabel('loss')
        plt.xlabel('iterations (per tens)')
        plt.title('learning rate = %f' % learning_rate)
        plt.show()
        
    return parameters

def model_mini_batch(X, Y, layers_dims, mini_batch_size=64, learning_rate=0.01, num_iterations=1500, print_cost=True, 
          initialize='random', figure=True):
    '''
    实现一个3层网络模型,采用mini_batch梯度下降法
    '''
    costs = []
    
    # step1 初始化参数
    if initialize == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialize == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialize == 'small':
        parameters = initialize_parameters_sqrt(layers_dims)
    else:
        print('初始化参数出错! 程序结束')
        exit
    
    # step2 开始学习
    seed = 1
    for i in range(num_iterations):
        seed = seed+1  # 使每次遍历数据集之后重新排序数据集
        mini_batchs = random_mini_batches(X, Y, mini_batch_size, seed)
        for mini_batch in mini_batchs:
            # 选择mini_batch
            (mini_batch_X, mini_batch_Y) = mini_batch
            # 向前传播
            A3, cache = forward_propagation(mini_batch_X, parameters)    
            # 计算损失
            loss = cross_entropy_loss(A3, mini_batch_Y)
            # 反向传播
            grads = backward_propagation(mini_batch_X, mini_batch_Y, cache)       
            # 更新参数
            parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 100 == 0:
            costs.append(loss)
            if print_cost:
                print('第 %d 次迭代，损失为%.3f' % (i, loss))
    if figure:
        plt.plot(np.squeeze(costs))
        plt.ylabel('loss')
        plt.xlabel('iterations (per tens)')
        plt.title('learning rate = %f' % learning_rate)
        plt.show()
        
    return parameters

def model_opt(X, Y, layers_dims, optimizer, initialize='random', mini_batch_size=64, beta1=0.9, beta2=0.999, epsilon=1e-8, 
              learning_rate=0.01, num_epochs=1500, print_cost=True, figure=True):
    '''
    实现一个3层网络模型,采用momentum, RMSProp, Adam梯度下降
    '''
    costs = []
    L = len(layers_dims)
    
    # step1 初始化参数
    if initialize == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialize == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialize == 'small':
        parameters = initialize_parameters_sqrt(layers_dims)
    else:
        print('初始化参数出错! 程序结束')
        exit
    # step2 选择优化器
    if optimizer == 'gd':
        pass
    elif optimizer == 'momentum':
        V = initilize_V(parameters)
    elif optimizer == 'RMSProp':
        S = initilize_S(parameters)
    elif optimizer == 'adam':
        V = initilize_V(parameters)
        S = initilize_S(parameters)
    else:
        print('optimizer参数出错，程序退出')
        exit(1)
    
    
    # step3 开始学习
    seed = 1
    t = 0 # 每学习完一个minibatch增加1
    for i in range(num_epochs):
        seed = seed+1  # 使每次遍历数据集之后重新排序数据集
        mini_batchs = random_mini_batches(X, Y, mini_batch_size, seed)
        for mini_batch in mini_batchs:
            # 选择mini_batch
            (mini_batch_X, mini_batch_Y) = mini_batch
            # 向前传播
            A3, cache = forward_propagation(mini_batch_X, parameters)    
            # 计算损失
            loss = cross_entropy_loss(A3, mini_batch_Y)
            # 反向传播
            grads = backward_propagation(mini_batch_X, mini_batch_Y, cache)       
            # 更新参数
            if optimizer == 'gd':
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == 'momentum':
                parameters, V = update_parameters_momentum(parameters, grads, V, beta1, learning_rate)
            elif optimizer == 'RMSProp':
                parameters, S = update_parameters_RMSProp(parameters, grads, S, beta2, learning_rate)
            elif optimizer == 'adam':
                t = t+1 # 更新t
                parameters, V, S = update_parameters_Adam(parameters, grads, V, S, beta1, beta2, t, learning_rate, epsilon)
            
        if i % 100 == 0:
            costs.append(loss)
            if print_cost:
                print('第 %d 次迭代，损失为%.3f' % (i, loss))
    if figure:
        plt.plot(np.squeeze(costs))
        plt.ylabel('loss')
        plt.xlabel('iterations (per tens)')
        plt.title('learning rate = %f' % learning_rate)
        plt.show()
        
    return parameters

def prediction(X, Y, parameters):
    '''
    该函数用于预测多层神经网络的结果
    '''
    m = Y.shape[1] # 训练样本的个数
    
    A3, cache = forward_propagation(X, parameters)
       
    Y_prediction = np.zeros((1, m))
    for i in range(m):
        if A3[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    accuracy = np.mean(Y_prediction[0,:] == Y[0,:])
    print('准确率为：%.3f%%'%(100*accuracy))  
    return Y_prediction
    

    
    
    
    
    
    
    
