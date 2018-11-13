# -*- coding: utf-8 -*-
"""
简介：
实现ResNet网络-----利用Keras实现
Created on Tue Nov 13 17:17:10 2018

@author: dell
"""
import resnets_utils   # 包含训练测试数据
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization
from keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_uniform



def identify_block(X, f, filters, stage, block):
    '''
    实现ResNet中的----The identify block
    Parameters:
        X - 输入的tensor类型的数据， shape=(m, n_H_prev, n_W_prev, n_C_prev)
        f - int型，指定了主路径上ConV窗口的维度
        filters - list, 定义了主路径上每层卷积层的filter的数目
        stage - int型，根据每层的位置来命名每一层，与block参数一起使用
        block - string型， 根据每层的位置来命名每一层， 与stage参数一起使用
    Returns:
        X - 恒等块的输出， tensor型， shape=(n_H, n_W, n_C)
    '''
    
    # 定义命名规则
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # 获取filter参数---卷积核的数目
    F1, F2, F3 = filters
    
    # 保存输入数据， 用于为主路径添加捷径
    X_shortcut = X
    
    # 主路径第一部分
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', 
               name=conv_name_base+'2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    # 主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same',
               name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    # 主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid',
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)
    
    # 将主路径上输出值和捷径上的值加在一起
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    '''
    实现ResNet中的-----The convolutional block
    
    Parameters:
        X - 输入的tensor， shape=(m, n_H_prev, n_W_prev, n_C_prev)
        f - int型， 卷积核的大小
        filters - list, 定义了主路径上每层卷积层的卷积核的数目
        stage - int型，根据每层的位置来命名每一层，与block参数一起使用
        block - string型， 根据每层的位置来命名每一层， 与stage参数一起使用
        s - strides操作移动的距离
    Returns:
        X - The convolutional block 的输出张量， shape=(n_H, n_W, n_C)
    '''
    
    # 定义命名规则
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # 获取每次卷积层上卷积核的大小
    F1, F2, F3 = filters
    
    # 保存输入数据用以添加到捷径上
    X_shortcut = X
    
    # 主路径第一部分
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # 主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    # 主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)
    
    # 捷径卷积操作部分
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    
    # 将主路径上输出值和捷径上的值加在一起
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape, classes):
    '''
    实现ResNet50
    Stage-1: CONV2D -> BATCHNORM -> RELU -> MAXPOOL
    Stage-2: CONVBLOCK -> IDBLOCK*2
    Stage-3: CONVBLOCK -> IDBLOCK*3
    Stage-4: CONVBLOCK -> IDBLOCK*5
    Stage-5: CONVBLOCK -> IDBLOCK*2
    Stage-6: AVGPOOL -> Flatten -> FC
    
    Parameters:
        input_shape - 输入图像数据的维度, shape=(n_H, n_W, n_C)
        classes - 分类数目
    Returns:
        model - ResNet50模型
    '''
    
    # 定义tensor类型的输入数据
    X_input = Input(shape=input_shape)
    
    # Zero-Padding
    X = ZeroPadding2D(padding=(3,3))(X_input)
    
    # Stage-1
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), name='conv1', 
               kernel_initializer=glorot_uniform(seed=0))(X)       
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3,3), strides=(2,2))(X)
    
    # Stage-2
    X = convolutional_block(X, f=3, filters=[64,64,256], stage=2, block='a', s=1)
    X = identify_block(X, f=3, filters=[64,64,256], stage=2, block='b')
    X = identify_block(X, f=3, filters=[64,64,256], stage=2, block='c')
    
    # Stage-3
    X = convolutional_block(X, f=3, filters=[128,128,512], stage=3, block='a', s=2)
    X = identify_block(X, f=3, filters=[128,128,512], stage=3, block='b')
    X = identify_block(X, f=3, filters=[128,128,512], stage=3, block='c')
    X = identify_block(X, f=3, filters=[128,128,512], stage=3, block='d')
    
    # Stage-4
    X = convolutional_block(X, f=3, filters=[256,256,1024], stage=4, block='a', s=2)
    X = identify_block(X, f=3, filters=[256,256,1024], stage=4, block='b')
    X = identify_block(X, f=3, filters=[256,256,1024], stage=4, block='c')
    X = identify_block(X, f=3, filters=[256,256,1024], stage=4, block='d')
    X = identify_block(X, f=3, filters=[256,256,1024], stage=4, block='e')
    X = identify_block(X, f=3, filters=[256,256,1024], stage=4, block='f')
    
    # Stage-5
    X = convolutional_block(X, f=3, filters=[512,512,2048], stage=5, block='a', s=2)
    X = identify_block(X, f=3, filters=[512,512,2048], stage=5, block='b')
    X = identify_block(X, f=3, filters=[512,512,2048], stage=5, block='c')
    
    # Stage-6
    X = Flatten()(X)
    X = Dense(units=classes, activation='softmax', name='fc'+str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet-50')
    
    return model


if __name__ == '__main__':
    X_train_org, Y_train_org, X_test_org, Y_test_org, classes = resnets_utils.load_dataset()
    
    # 数据归一化
    X_train = X_train_org / 255
    X_test = X_test_org / 255
    
    # 将标签数据转化为one-hot类型
    Y_train = resnets_utils.convert_to_one_hot(Y_train_org, 6).T
    Y_test = resnets_utils.convert_to_one_hot(Y_test_org, 6).T
    
    # 创建一个模型
    input_shape = X_train.shape[1:]
    ResNet = ResNet50(input_shape=input_shape, classes=6)
    
    # 编译模型
    ResNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 训练模型
    ResNet.fit(X_train, Y_train, validation_split=0.2, epochs=2, batch_size=32, verbose=2)
    
    # 评估模型
    score = ResNet.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    

    
 
    
    