# -*- coding: utf-8 -*-
"""
简介：
利用tensorflow 的多层全连接网络分类
Created on Fri Dec  7 19:34:57 2018

@author: dell
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils

import time

tf.reset_default_graph()

# define parameters for the model
learning_rate=0.01
batch_size = 128
n_epochs = 30

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 10 # 3nd layer number of neurons
n_x = 784 # MNIST data input (img shape: 28*28)
n_y = 10 # MNIST total classes (0-9 digits)


# step1 read in data
def load_data():
    
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    
    train_X = train_X.reshape((len(train_X), -1))
    test_X = test_X.reshape(len(test_X), -1)
    
    train_Y = np_utils.to_categorical(train_Y, 10)
    test_Y = np_utils.to_categorical(test_Y, 10)
    
    return train_X, train_Y, test_X, test_Y


def creat_placeholders(n_x, n_y):
    '''
    为tensorflow创建占位符
    Parameters:
        n_x - 图片向量的大小
        n_y - 分类数
    Returns:
        X - 输入数据的占位符，维度(None, n_x)
        Y - 输入标签的占位符，维度(None, n_y)
    Note:
        使用None可以让我们灵活的处理占位符提供的样本
    '''
    
    X = tf.placeholder(tf.float32, shape=[None, n_x], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, n_y], name='Y')
    
    return X, Y

def initialize_parameters():
    '''
    初始化神经网络的参数
    Returns:
        parameters - 包含W和b的字典
    '''
    
    tf.set_random_seed(1) # 指定随机种子
    weights = {
            'W1':tf.get_variable('W1', shape=[n_x, n_hidden_1], initializer=tf.contrib.layers.xavier_initializer(seed=1)),
            'W2':tf.get_variable('W2', shape=[n_hidden_1, n_hidden_2], initializer=tf.contrib.layers.xavier_initializer(seed=1)),
            'W3':tf.get_variable('W3', shape=[n_hidden_2, n_hidden_3], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            }
    
    biases = {
            'b1':tf.get_variable('b1', shape=[n_hidden_1], initializer=tf.zeros_initializer()),
            'b2':tf.get_variable('b2', shape=[n_hidden_2], initializer=tf.zeros_initializer()),
            'b3':tf.get_variable('b3', shape=[n_hidden_3], initializer=tf.zeros_initializer())
            }
   
    

    return weights, biases

def forward_propagation(X, weights, biases):
    '''
    实现一个模型的向前传播，三个隐藏层
    Parameters:
        X - 输入数据集
        weights - 初始化的参数
        biases - 初始化的参数
    Returns:
        Z3 - 最后一层的线性输出
    '''
    
 
    Z1 = tf.add(tf.matmul(X, weights['W1']), biases['b1'])
    A1 = tf.nn.relu(Z1)
    
    Z2 = tf.add(tf.matmul(A1, weights['W2']), biases['b2'])
    A2 = tf.nn.relu(Z2)
    
    Z3 = tf.add(tf.matmul(A2, weights['W3']), biases['b3'])
    
    return Z3

def compute_cost(Z3, Y):
    '''
    计算损失函数
    Parameters:
        Z3 - 最后一层的线性输出矩阵[6, Zone]
    Returns:
        cost - 损失值
    '''
    
    logits = Z3   #计算Z3的转置--------------------------------??
    labels = Y
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost

def model(train_X, train_Y, test_X, test_Y, learning_rate=learning_rate, num_epochs=n_epochs,
          batch_size=batch_size, figure=True):
    '''
    实现一个三层的tensorflow神经网络
    Returns:
        parameters - 训练后的参数
    '''
    tf.reset_default_graph()   #能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)
    seed = 3
    
    costs = []
    
    # step1 为X和Y创建placeholder
    X, Y = creat_placeholders(n_x, n_y)
    
    # step2 初始化参数W和b
    weights, biases = initialize_parameters()
    
    # step3 向前传播
    Z3 = forward_propagation(X, weights, biases)
    
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
        n_batches = 500       
        for i in range(num_epochs):
            total_loss = 0
            
            for j in range(n_batches):
                indices = np.random.randint(low=0, high=len(train_X), size=batch_size)
                X_batch, Y_batch = train_X[indices], train_Y[indices]
                _, batch_loss = sess.run([optimizer, cost], feed_dict={X:X_batch, Y:Y_batch})
                total_loss += batch_loss
            epoch_loss = total_loss / n_batches   
            print('Average loss epochs {0}: {1}'.format(i, epoch_loss))
            costs.append(epoch_loss)
        if figure:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per epoch)')
            plt.title('learning_rate = ' +str(learning_rate))
        plt.show()
        
        
        # 保存训练后的参数
        weights_out, biases_out = sess.run([weights, biases])
        
        # 计算当前的预测结果
        pred = Z3
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        
        # 计算正确率 tf.cast--数据类型转换函数
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        print('训练集的准确率：', accuracy.eval({X:train_X, Y:train_Y})) # --------???
        print('测试集的准确率：', accuracy.eval({X:test_X, Y:test_Y}))

train_X, train_Y, test_X, test_Y = load_data()
model(train_X, train_Y, test_X, test_Y, learning_rate=learning_rate, num_epochs=n_epochs,
          batch_size=batch_size, figure=True)