# -*- coding: utf-8 -*-
"""
简介：
基于tensorflow的多层神经网络softmax分类(二分类)

Created on Mon Nov  5 18:59:14 2018

@author: GEAR
"""

import numpy as np
import tf_utils
import init_utils
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

def creat_placeholders(n_x, n_y):
    '''
    为tensorflow创建占位符
    Parameters:
        n_x - 输入数据维数
        n_y - 对应标签维数
    Returns:
        X - 输入数据的占位符，维度(n_x, None)
        Y - 输入标签的占位符，维度(n_y, None)
    Note:
        使用None可以让我们灵活的处理占位符提供的样本
    '''
    
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')
    
    return X, Y

def initialize_parameters(layers_dims):
    '''
    初始化神经网络的参数，采用xaiver初始化
    Returns:
        parameters - 包含W和b的字典
    '''
    
    parameters = {}
    L = len(layers_dims)    #网络总层数
    tf.set_random_seed(1) # 指定随机种子
    for i in range(1, L):
        parameters['W'+str(i)] = tf.get_variable('W'+str(i), shape=[layers_dims[i], layers_dims[i-1]], 
                                   initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters['b'+str(i)] = tf.get_variable('b'+str(i), shape=[layers_dims[i], 1], initializer=tf.zeros_initializer())
       
    return parameters

def forward_propagation(X, parameters):
    '''
    实现一个模型的向前传播，三个隐藏层
    Parameters:
        X - 输入数据集
        parameters - 初始化的参数
    Returns:
        Z3 - 最后一层的线性输出
    '''
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    
    return Z3

def compute_cost(Z3, Y):
    '''
    计算损失函数
    Parameters:
        Z3 - 最后一层的线性输出矩阵[2, Zone]
    Returns:
        cost - 损失值
    '''
    
    logits = tf.transpose(Z3)   #计算Z3的转置--------------------------------??
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost

def model(train_X, train_Y, test_X, test_Y, layers_dims, learning_rate=0.003, num_epochs=1500,
          minibatch_size=128, print_cost=True, figure=True):
    '''
    实现一个三层的tensorflow神经网络
    Returns:
        parameters - 训练后的参数
    '''
    ops.reset_default_graph()   #能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = train_X.shape
    n_y = train_Y.shape[0]
    costs = []
    
    # step1 为X和Y创建placeholder
    X, Y = creat_placeholders(n_x, n_y)
    
    # step2 初始化参数W和b
    parameters = initialize_parameters(layers_dims)
    
    # step3 向前传播
    Z3 = forward_propagation(X, parameters)
    
    # step4 计算损失函数
    cost = compute_cost(Z3, Y)
    
    # step5 反向传播, 采用Adam算法
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # step6 初始化所有变量
    init = tf.global_variables_initializer()
    
    # step7 开始计算
    with tf.Session() as sess:
        #初始化
        sess.run(init)
        # 开始训练
        for epoch in range(num_epochs):
            epoch_cost = 0  #每次迭代的损失
            num_minibatchs = int(m/minibatch_size)  # minibatch的总数目
            seed = seed+1
            minibatchs = tf_utils.random_mini_batches(train_X, train_Y, minibatch_size, seed)
            
            for minibatch in minibatchs:
                (minibatch_X, minibatch_Y) = minibatch
                
                # 开始运行session
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                # 计算这个minibatch在这一代中所占的误差
                epoch_cost = epoch_cost +minibatch_cost / num_minibatchs
                
            if epoch % 5 ==0:
               costs.append(epoch_cost)
               if print_cost and epoch % 100 ==0:
                   print('epoch= ' + str(epoch) + '     epoch_cost= ' + str(epoch_cost))
        if figure:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title('learning_rate = ' +str(learning_rate))
        plt.show()
        
        
        # 保存训练后的参数
        parameters = sess.run(parameters)
        
        # 计算当前的预测结果
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        
        # 计算正确率 tf.cast--数据类型转换函数
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        
        print('训练集的准确率：', accuracy.eval({X:train_X, Y:train_Y})) # --------???
        print('测试集的准确率：', accuracy.eval({X:test_X, Y:test_Y}))
        
        return parameters


def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = tf_utils.forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction
    


# 载入数据
train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=True)
train_Y = tf_utils.convert_to_one_hot(train_Y, 2)
test_Y = tf_utils.convert_to_one_hot(test_Y, 2)
layers_dims = [train_X.shape[0], 10, 5, 2]  #使用softmax时输出有2个对二分类
plt.show()


#开始训练
parameters = model(train_X, train_Y, test_X, test_Y, layers_dims, learning_rate=0.001)


