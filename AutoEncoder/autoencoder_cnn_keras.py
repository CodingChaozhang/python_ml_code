# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:58:52 2018

@author: dell
"""

import numpy as np
from keras.datasets import mnist
from keras import Model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D
import matplotlib.pyplot as plt

np.random.seed(1)


def AutoEncoder(input_shape):
    '''
    实现一个卷积(CNN)自编码器(AutoEncoder)
    Parameters:
        input_shape -- 输入的数据维度,如输入的一张28*28图片，shape=(28, 28, 1)
    Returns:
        model -- 创建的AutoEncoder模型
    '''
    
    # 输入数据的shape
    encoder_input = Input(shape=(input_shape))
    encoder_dim = 2
    
    # 建立编码层
    encoded = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(encoder_input)
    encoded = MaxPool2D(pool_size=(2,2), padding='same')(encoded)
    
    encoded = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(encoded)
    encoder_output = MaxPool2D(pool_size=(2,2), padding='same')(encoded)
    
      
    # 建立解码层
    decoded = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(encoder_output)
    decoded = UpSampling2D(size=(2,2))(decoded)
      
    decoded = Conv2D(filters=16, kernel_size=(3,3),activation='relu', padding='same')(decoded)
    decoded = UpSampling2D(size=(2,2))(decoded)
    
    decoded = Conv2D(filters=1, kernel_size=(3,3), activation='tanh', padding='same')(decoded)
     
    # 建立自编码模型
    autoencoder = Model(inputs=encoder_input, outputs=decoded)
    
    autoencoder.summary()
    
    # 建立编码器模型
#    encoder = Model(inputs=encoder_input, outputs=encoder_output)
    
    return autoencoder

if __name__ == '__main__':
    # 载入数据        
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    # 数据预处理
    X_train = X_train.astype(np.float32)/255. - 0.5         # 将数据映射到-0.5 -- 0.5
    X_test = X_test.astype(np.float32)/255. - 0.5
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    # 创建一个模型
    input_shape = (28, 28, 1)
    autoencoder = AutoEncoder(input_shape)
    
    # 编译模型
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    # 训练模型
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, 
                    validation_split=0.2, verbose=2)
    

    # autoencoder后的数据
    autoencoder_imgs =  autoencoder.predict(X_test)
    n = 10 #显示的图片数量
    plt.figure(figsize=(20,4))
    
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(X_test[i].reshape(28,28))
        plt.gray()
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(autoencoder_imgs[i].reshape(28,28))
        plt.gray()
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()

 