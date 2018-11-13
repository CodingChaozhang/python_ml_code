# -*- coding: utf-8 -*-
"""
简介：
采用Keras框架实现识别人笑脸的网络
Created on Mon Nov 12 19:53:45 2018

@author: gear
"""

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPool2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
#K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
import kt_utils


def HappyModel(input_shape):
    '''
    实现一个检测型
    Parameters:
        input_shape: 输入的数据维度,这里一般只是输入的一张图片，shape=(n_H, n_W, n_C)
    Returns:
        model: 创建的Keras模型
    '''
    # 输入的训练数据
    X_input = Input(input_shape)
    
    # 使用0填充
    X = ZeroPadding2D(padding=(3, 3))(X_input)
    
    # 对X使用CONV -> BN -> RELU
    X = Conv2D(filters=32, kernel_size=(7,7), strides=(1,1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)   # 这里默认axis=1
    X = Activation('relu')(X)
    
    # 加入池化层1
    X = MaxPooling2D(pool_size=(2,2), name='max_pool')(X)
    
    #降维加入全连接层
    X = Flatten()(X)
    X = Dense(units=1, activation='sigmoid', name='fc')(X)
    
    # 创建模型，我们将对其进行训练测试
    model = Model(inputs=X_input, outputs=X, name='HappyModel')
    
    return model

def my_loss(y_true, y_pred):
    '''
    自定义损失函数
    '''
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

def my_loss2(y_true, y_pred):
    '''
    自定义损失函数
    '''
    cost = -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
    
    return cost

if __name__ == '__main__':
    train_X_org, train_Y_org, test_X_org, test_Y_org, classes = kt_utils.load_dataset()
    
    # step1 对数据进行归一化处理
    train_X = train_X_org / 255
    test_X = test_X_org / 255
    
    # step2 对标签数据进行转置处理，在keras中采用(样本数， 标签)这样的输入！！！
    train_Y = train_Y_org.T
    test_Y = test_Y_org.T
    
    # step3 创建一个模型
    input_shape = train_X.shape[1:]         # 64 X 64 X 3
    happy_model = HappyModel(input_shape)
    
    # step4 编译模型（定义训练方式）
#    happy_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    happy_model.compile(loss=my_loss2, optimizer='adam', metrics=['accuracy'])
    
    # step5 训练模型
    happy_model.fit(train_X, train_Y, validation_split=0.2, epochs=40, batch_size=50, verbose=2)
    
    # step6 评估模型
    score = happy_model.evaluate(test_X, test_Y, batch_size=32, verbose=1, sample_weight=None)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    
    
    
    
    





