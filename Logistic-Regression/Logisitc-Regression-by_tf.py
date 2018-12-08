# -*- coding: utf-8 -*-
"""
简介
CS20-使用placeholder方式建立计算图--logreg
Created on Fri Dec  7 11:38:30 2018

@author: gear
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

# step1 read in data
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

train_X = train_X.reshape((len(train_X), -1))
test_X = test_X.reshape(len(test_X), -1)

train_Y = np_utils.to_categorical(train_Y, 10)
test_Y = np_utils.to_categorical(test_Y, 10)

# step2 create placeholders for X, Y
X = tf.placeholder(tf.float32, shape=[None, 784], name='image')
Y = tf.placeholder(tf.float32, shape=[None, 10], name='label')

# step3 create weights and bias
# shape of w depends on the dimension of X and Y
# shape of b depends on Y
w = tf.get_variable(name='weight', shape=[784, 10], initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=[1, 10], initializer=tf.zeros_initializer())

# step4 build the model
logits = tf.matmul(X, w) + b

# step5 create the loss function

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)

# step6 using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# step7 calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
#tf.cast(x, dtype, name=None),将x的数据格式转化成dtype
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg_placeholder', tf.get_default_graph())

# step8 train the model
start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_batches = 500
    
    for i in range(n_epochs):
        total_loss = 0
        
        for j in range(n_batches):
            indices = np.random.randint(low=0, high=len(train_X), size=batch_size)
            X_batch, Y_batch = train_X[indices], train_Y[indices]
            _, batch_loss = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch})
            total_loss += batch_loss
            
        print('Average loss epochs {0}: {1}'.format(i, total_loss / n_batches))
    end = time.time()
    print('Total time: {0} seconds'.format(end - start))
    
    # step9 test the model
    accuracy = sess.run(accuracy, feed_dict={X:test_X, Y:test_Y})
        
    print('Accuracy {0}'.format(accuracy))
    
writer.close()


