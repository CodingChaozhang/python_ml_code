# -*- coding: utf-8 -*-
"""
tensorflow基础操作程序
Created on Mon Nov  5 16:16:13 2018

@author: GEAR
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

'''
-----------------------------case1 计算线性函数-----------------------------
'''
y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')

loss = tf.Variable((y - y_hat)**2, name='loss')

init = tf.global_variables_initializer()    # 运行之后的初始化

with tf.Session() as session:
    print(session.run(init)) #初始化变量
    print(session.run(loss))    #显示计算结果
    
#    
#'''
#-----------------------------case1 计算Y = WX + b------------------------------
#'''
#def linear_function():
#    '''
#    实现一个线性函数计算的功能
#    '''
#    np.random.seed(1)
#    
#    X = np.random.randn(3,1)
#    W = np.random.randn(4,3)
#    b = np.random.randn(4,1)
#    
#    Y = tf.add(tf.matmul(W, X), b)
##    Y = tf.matmul(W, X) + b
#    
#    # 创建一个session兵运行
#    with tf.Session() as session:
#        result = session.run(Y)
#        
#    return result
#
#'''
#-----------------------------case1 计算sigmoid函数------------------------------
#'''
#def sigmoid(z):
#    '''
#    实现sigmoid函数
#    '''
#    
#    # 创建一个占位符x
#    x = tf.placeholder(tf.float32, name='x')
#    
#    # 计算sigmoid函数
#    sigmoid = tf.sigmoid(x)
#    
#    # 创建一个会话
#    with tf.Session() as sess:
#        result = sess.run(sigmoid, feed_dict={x:z})
#    
#    return result
#
#'''
#-----------------------------case1 计算one-hot encoding------------------------------
#'''
#
#def one_hot_matrix(labels, C):
#    '''
#    创建一个one-hot编码矩阵
#    Parameters:
#        labels - 标签向量（1,m)
#        C - 类别数
#    Returns:
#        one_hot - 转化后的one-hot矩阵
#    '''
#    
#    # 创建一个tf.constant 
#    C = tf.constant(C, name='C')
#    # 使用tf.one_hot,注意axis=0扩展为列向量
#    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
#    # 创建一个session
#    with tf.Session() as sess:
#        one_hot = sess.run(one_hot_matrix)
#        
#    return one_hot
# 
#labels = [1,2,3,0,2,1]
#one_hot = one_hot_matrix(labels, C=4)
#print(one_hot)


    

