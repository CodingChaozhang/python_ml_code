# -*- coding: utf-8 -*-
"""
简介：采用tensorflow建立多层卷积网络识别手势
Created on Sat Nov 10 16:54:12 2018

@author: gear
"""
import cnn_utils
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    '''
    创建占位符
    Parameters:
        n_H0 - 输入图像的height
        n_W0 - 输入图像的width
        n_C0 - 输入图像的channel
        n_y - 分类的数目
    Returns:
        X - 输入数据的占位符， shape=(None, n_H0, n_W0, n_C0)
        Y - 输入标签的占位符， shape=(None, n_y)
    '''
    X = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, shape=[None, n_y])
    
    return X, Y


def initialize_parameters():
    '''
    初始化卷积核(权重矩阵)
    Returns:
        parameters - 包含各层卷积核的字典
    '''
    W1 = tf.get_variable('W1', shape=[4,4,3,8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2', shape=[2,2,8,16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    parameters = {'W1':W1,
                  'W2':W2}
    
    return parameters

def forward_propagation(X, parameters):
    '''
    实现CNN的向前传播算法： 
    Conv2D -> Relu -> MaxPooling -> Conv2D -> Relu -> MaxPooling -> Flatien -> FullyConnected  
    
    Parameters:
        X - 输入的图像数据集，shape=(m, n_H0, n_W0, n_C0)
        parameters - 包含初始化卷积核的字典
    Returns:
        Z3 - 最后一层全连接层的线性输出
    '''
    # step1 从字典中读取初始化的卷积核W1和W2
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    # step2 第一层卷积运算: f=4, strides=1
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    # step3 将卷积后的值输入激活函数
    A1 = tf.nn.relu(Z1)
    # step4 进行第一次maxpooling运算: f=8, strides=8
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
    # step5 第二层卷积运算: f=2, strides=1
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    # step6 将卷积后的值输入激活函数
    A2 = tf.nn.relu(Z2)
    # step7 进行第二次maxpooling运算: f=4, strides=4
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
    # step8 将卷积层展开为全连接层
    P = tf.contrib.layers.flatten(P2)
    # step9 加入一层全连接层，总共有6个神经元
    Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)    # 这里没有激活函数
    
    return Z3

def compute_cost(Z3, Y):
    '''
    计算损失函数
    Parameters:
        Z3 - 最后一层全连接层的线性输出， shape=(6, 样本数)
        Y - 训练数据对应的样本标签，shape=(6, 样本数)
    Returns:
        cost - 损失的张量
    '''
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    
    return cost
    
def model(train_X, train_Y, test_X, test_Y, learning_rate=0.09, num_epochs=100, 
          minibatch_size=64, print_cost=True):
    '''
    实现一个有两个卷积层的CNN网络
    Parameters:
        train_X - shape=(None, 64, 64, 3)
        train_Y - shape=(None, n_y=6)
    Returns:
        train_accuracy - 训练集上的准确率
        test_accuracy - 测试集上的准确率
        parameters - 模型学习到的权重参数的字典
    '''
    
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = train_X.shape
    n_y = train_Y.shape[1]
    costs = []
    
    # step1 创建Ｘ和Ｙ的占位符
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    # step2 卷积核初始化
    parameters = initialize_parameters()
    # step3 向前传播
    Z3 = forward_propagation(X, parameters)
    # step4 计算损失函数
    cost = compute_cost(Z3, Y)
    # step5 反向传播
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # step6 初始化所有变量
    init = tf.global_variables_initializer()
    
    # step7开始计算tensorflow计算图
    with tf.Session() as sess:
        sess.run(init)
        # 开始训练
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = cnn_utils.random_mini_batches(train_X, train_Y, minibatch_size, seed)
            
            for minibatch in minibatches:
                # 选取一个minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # 开始计算对应的损失
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                
            if print_cost and epoch % 5 == 0:
                print('Cost after epoch %d: %.3f' % (epoch, minibatch_cost))
            if print_cost and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        plt.plot(np.squeeze(costs))
        plt.xlabel('iterations (per tens)')
        plt.ylabel('cost')
        plt.title('Learning rate =' + str(learning_rate))
        plt.show()
        
        # 计算预测的正确率
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # 计算在train_Set 和 test_Set上的准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy)
        train_accuracy = accuracy.eval({X:train_X, Y:train_Y})
        test_accuracy = accuracy.eval({X:train_X, Y:train_Y})
        print('Train Accuracy:', train_accuracy)
        print('Test Accuracy:', test_accuracy)
        
        return train_accuracy, test_accuracy, parameters

    
if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y, classes = cnn_utils.load_dataset()
    # 归一化数据集
    train_X = train_X / 255
    test_X = test_X / 255
    # 将标签转换为one-hot编码
    train_Y = cnn_utils.convert_to_one_hot(train_Y, 6).T
    test_Y = cnn_utils.convert_to_one_hot(test_Y, 6).T
    
    _, _, parameters = model(train_X, train_Y, test_X, test_Y, num_epochs=100)
    
    