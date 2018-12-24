# -*- coding: utf-8 -*-
"""简介：
线性二分类SVM---by tensorflow
Created on Mon Dec 24 15:10:53 2018

@author: gear
"""

import numpy as np
import tensorflow as tf

from sklearn import datasets
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

################parameters################
C_param = 0.1
Reg_param = 1.0
n_epochs = 500
batch_size = 32
lr = 0.1
delta = 1.0




def load_data():
    
    iris = datasets.load_iris()    
    X = iris.data[:, :2]
    Y = np.array([1 if label==0 else -1 for label in iris.target])
    
    index = [i for i in range(len(X))]
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]
    
    train_X = X[0:100]
    train_Y = Y[0:100].reshape(-1,1)
    
    test_X = X[100:]
    test_Y = Y[100:].reshape(-1,1)
    
    return train_X, train_Y ,test_X, test_Y

def creat_placeholder():
    
    X = tf.placeholder(tf.float32, shape=[None, 2])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    
    return X, Y

def initialize_parameters():
    
    W = tf.get_variable(name='weights', shape=[X.shape[1], 1], initializer=tf.random_normal_initializer)
    b = tf.get_variable(name='bias', shape=[1,1], initializer=tf.random_normal_initializer)
    
    return W, b

def loss_fn(W, b, X, Y):
    logit = tf.add(tf.matmul(X, W), b)
    norm_term = tf.divide(tf.reduce_sum(tf.multiply(tf.transpose(W), W)), 2)
    classify_loss = tf.reduce_mean(tf.maximum(0., tf.subtract(delta, tf.multiply(logit, Y))))    
    total_loss = tf.multiply(C_param, classify_loss) + tf.multiply(Reg_param, norm_term)
    
    return total_loss

def prediction(W, b, X, Y):
    predicts = tf.sign(tf.matmul(X, W) + b)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, Y), tf.float32))
    
    return accuracy, predicts


def SVM():
    ops.reset_default_graph()
    
    train_X, train_Y ,test_X, test_Y = load_data()
    
    X, Y = creat_placeholder()
    
    W, b = initialize_parameters()
    
    loss = loss_fn(W, b, X, Y)
    
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    
    init = tf.initialize_all_variables()
    costs = []
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(n_epochs):
            total_loss = 0
            n_batches = 20        
            for batch in range(n_batches):
                indices = np.random.randint(low=0, high=len(train_X), size=batch_size)
                X_batch, Y_batch = train_X[indices], train_Y[indices]
                _, batch_loss = sess.run([opt, loss], feed_dict={X:X_batch, Y:Y_batch})
                total_loss += batch_loss
            epoch_loss = total_loss / n_batches   
            print('Average loss epochs {0}: {1}'.format(epoch, epoch_loss))
            costs.append(epoch_loss)
    
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per epoch)')
        plt.title('learning_rate = ' +str(lr))
        plt.show()
                
        
        accuracy, pred = prediction(W, b, X, Y)
        print('测试集的准确率：', accuracy.eval({X:test_X, Y:test_Y}))
        pred = sess.run(pred, feed_dict={X:test_X})
    

    return pred



###################################--main_code--###############################
train_X, train_Y ,test_X, test_Y = load_data()
pred = SVM()

