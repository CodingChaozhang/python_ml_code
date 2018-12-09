# -*- coding: utf-8 -*-
"""
keras --function_API -- CNN model
Created on Sun Dec  9 15:23:03 2018

@author: gear
"""

import numpy as np
import keras
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

def CNN(input_shape):
    '''
    Parameters:
        shape -- 输入的数据维度
    Returns:
        model -- 创建的ANN模型
    '''
    # 输入的训练数据
    X_input = keras.layers.Input(input_shape)
    
    # 使用0填充
    X = keras.layers.ZeroPadding2D(padding=(3, 3))(X_input)
    
    # 对X使用CONV -> BN -> RELU
    X = keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), name='conv0')(X)
    X = keras.layers.BatchNormalization(axis=3, name='bn0')(X)   # 这里默认axis=1
    X = keras.layers.Activation('relu')(X)
    
    # 加入池化层1
    X = keras.layers.MaxPooling2D(pool_size=(2,2), name='max_pool')(X)
    
    #降维加入全连接层
    X = keras.layers.Flatten()(X)
    output = keras.layers.Dense(units=10, activation='softmax', name='fc')(X)
    
    # 创建模型，我们将对其进行训练测试
    model = keras.models.Model(inputs=X_input, outputs=output, name='CNN_Model')
    
    return model


def loss()

def TrainCNN(model, train_X, train_Y, test_X, test_Y):
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
    
    # step1 read in data
    train_X, train_Y, test_X, test_Y = load_data()
    
    # step2 bulid the model
    input_shape=train_X.shape[1:]
    img_model = CNN(input_shape)
    
    # step3 train the model
    pred_Y = TrainCNN(img_model, train_X, train_Y, test_X, test_Y)
        
    