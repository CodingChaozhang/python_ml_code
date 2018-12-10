# -*- coding: utf-8 -*-
"""

简介：采用tensorflow建立多层卷积网络识别mnist
Created on Sat Dec  8 20:47:16 2018

@author: dell
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils

import time

def load_data():
    
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    
    train_X = train_X.reshape((-1, 28, 28, 1))
    test_X = test_X.reshape((-1, 28, 28, 1))
    train_Y = np_utils.to_categorical(train_Y, 10)
    test_Y = np_utils.to_categorical(test_Y, 10)
    
    return train_X, train_Y, test_X, test_Y


def create_placeholders(n_h, n_w, n_c, n_y):
    '''
    创建占位符
    Parameters:
        n_H0 - 输入图像的height
        n_W0 - 输入图像的width
        n_C0 - 输入图像的channel
        n_y - 分类的数目
    Returns:
        X - 输入数据的占位符， shape=(None, n_H, n_W, n_C)
        Y - 输入标签的占位符， shape=(None, n_y)
    '''
    X = tf.placeholder(tf.float32, shape=[None, n_h, n_w, n_c])
    Y = tf.placeholder(tf.float32, shape=[None, n_y])
    
    return X, Y

def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    '''
    建立激活函数是relu的卷积层函数
    Parameters:
        inputs -- 输入卷积层的数据
    Returns:
        激活函数的值
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]  # 输入图像的通道数
        kernel = tf.get_variable(name='kernel', 
                                 shape=[k_size, k_size, in_channels, filters], 
                                 initializer=tf.truncated_normal_initializer())
        
        biases = tf.get_variable(name='biases', 
                                 shape=[filters],
                                 initializer=tf.random_normal_initializer())
        
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
        return tf.nn.relu(conv + biases, name=scope.name)
    

def maxpooling(inputs, k_size, stride, padding='VALID', scope_name='pool'):
    '''
    建立maxpooling层的函数
    
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs,
                              ksize=[1, k_size, k_size, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding)
        return pool
    
def fully_connected(inputs, out_dim, scope_name='fc'):
    '''
    建立全连接层函数
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable(name='weights',
                            shape=[in_dim, out_dim], 
                            initializer=tf.truncated_normal_initializer())
        b =tf.get_variable(name='biases', 
                           shape=[out_dim], 
                           initializer=tf.constant_initializer(0.0))
        
        out = tf.matmul(inputs, w) + b
        
    return out

def ConNet(inputs, keep_prob, n_classes):
    '''
    建立网络
    '''
    keep_prob = tf.constant(keep_prob)       # 这里直接输入0.75会出错
    conv1 = conv_relu(inputs= inputs,
                      filters=32,
                      k_size=5,
                      stride=1,
                      padding='SAME',
                      scope_name='conv1')
    
    pool1 = maxpooling(inputs=conv1,
                       k_size=2,
                       stride=2,
                       padding='VALID',
                       scope_name='pool1')
    
    pool1 = tf.layers.flatten(pool1)
    
#    fc = fully_connected(inputs=pool1,
#                         out_dim=1024,
#                         scope_name='fc')
    fc = tf.layers.dense(pool1, 1024, activation=tf.nn.relu, name='fc')

    logits = fully_connected(fc, n_classes, 'logits')
    
    return logits

def compute_loss(logits, Y):
    '''
    损失函数定义
    '''
    with tf.name_scope('loss'):
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits=logits)
        loss = tf.reduce_mean(entropy, name='loss')
        
    return loss

def optimizer(learning_rate, loss):
    '''
    定义训练的方法
    '''
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    return opt
    

def evaluation(logits, Y):
    with tf.name_scope('predict'):
        preds = tf.nn.softmax(logits)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    
    return accuracy
 
       
def ConvNetmodel(train_X, train_Y, test_X, test_Y, learning_rate=0.001, num_epochs=30, 
          batch_size=128, print_cost=True):
    '''
    实现一个有两个卷积层的CNN网络
    Parameters:
        train_X - shape=(None, 28, 28, 3)
        train_Y - shape=(None, n_y=10)
    Returns:
        test_accuracy - 测试集上的准确率
        parameters - 模型学习到的权重参数的字典
    '''
    
    tf.reset_default_graph()
    (m, n_H, n_W, n_C) = train_X.shape
    n_y = train_Y.shape[1]
    costs = []
    
    # step1 创建Ｘ和Ｙ的占位符
    X, Y = create_placeholders(n_H, n_W, n_C, n_y)
    
    # step2 向前传播
    logits = ConNet(inputs=X, keep_prob=0.75, n_classes=n_y)
    
    # step3 计算损失函数
    loss = compute_loss(logits, Y)
    
    # step4 反向传播
    opt = optimizer(learning_rate, loss)
    
    # step5 初始化所有变量
    init = tf.global_variables_initializer()

    # step6开始计算tensorflow计算图
    with tf.Session() as sess:
        #初始化
        sess.run(init)
        # 开始训练
        n_batches = 20      
        for i in range(num_epochs):
            total_loss = 0
            
            for j in range(n_batches):
                indices = np.random.randint(low=0, high=len(train_X), size=batch_size)
                X_batch, Y_batch = train_X[indices], train_Y[indices]
                _, batch_loss = sess.run([opt, loss], feed_dict={X:X_batch, Y:Y_batch})
                total_loss += batch_loss
            epoch_loss = total_loss / n_batches   
            print('Average loss epochs {0}: {1}'.format(i, epoch_loss))
            costs.append(epoch_loss)
    
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per epoch)')
        plt.title('learning_rate = ' +str(learning_rate))
        plt.show()
        
               
        # 计算当前的预测结果
        accuracy = evaluation(logits, Y)
        print('测试集的准确率：', accuracy.eval({X:test_X, Y:test_Y}))
        
        

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_data()
    ConvNetmodel(train_X, train_Y, test_X, test_Y)
        
        
        
                                   



        
        
        

    
