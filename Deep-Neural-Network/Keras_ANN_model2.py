# -*- coding: utf-8 -*-
"""
简介：
利用keras的多层全连接网络分类
Created on Fri Dec  7 21:05:06 2018

@author: dell
"""

import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils

import time


def load_data():
    
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    
    train_X = train_X.reshape((len(train_X), -1))
    test_X = test_X.reshape(len(test_X), -1)
    
    train_Y = np_utils.to_categorical(train_Y, 10)
    test_Y = np_utils.to_categorical(test_Y, 10)
    
    return train_X, train_Y, test_X, test_Y

def ANN():
    '''
    Parameters:
        shape -- 输入的数据维度
    Returns:
        model -- 创建的ANN模型
    '''
    X_input = keras.layers.Input(shape=(n_x, ))
    
    # 01 layer
    X = keras.layers.Dense(units=n_hidden_1, activation='relu')(X_input)
    # 02 layer
    X = keras.layers.Dense(units=n_hidden_2, activation='relu')(X)
    # 03 layer
    output = keras.layers.Dense(units=n_hidden_3, activation='softmax')(X)
    
    model = keras.models.Model(inputs=X_input, outputs=output)
    model.summary()
    
    return model

def TrainANN(model, train_X, train_Y, test_X, test_Y):
    '''
    训练刚才建立的ANN模型
    Parameter:
        model -- 建立的模型
        train_X -- 训练数据
        train_Y -- 训练标签
    Returns:
        None
    '''
    # step1 定义训练方式
    model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
    
    # step2 开始训练
    model.fit(train_X, train_Y, epochs=n_epochs, batch_size=batch_size)
    
    # step3 评估模型
    score = model.evaluate(test_X, test_Y)
    
    # step4 进行预测
    # model.predict_classes 不能用于函数式模型
    pred_Y = model.predict(test_X)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return pred_Y



if __name__ == '__main__':
    # define parameters for the model
    batch_size = 256
    n_epochs = 30
    
    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of neurons
    n_hidden_2 = 256 # 2nd layer number of neurons
    n_hidden_3 = 10 # 3nd layer number of neurons
    n_x = 784 # MNIST data input (img shape: 28*28)
    n_y = 10 # MNIST total classes (0-9 digits)
    
    # step1 read in data
    train_X, train_Y, test_X, test_Y = load_data()
    
    # step2 bulid the model
    img_model = ANN()
    
    # step3 train the model
    pred_Y = TrainANN(img_model, train_X, train_Y, test_X, test_Y)
        
    