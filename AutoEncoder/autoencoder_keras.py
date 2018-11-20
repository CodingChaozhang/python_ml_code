# -*- coding: utf-8 -*-
"""
简介：
autoEncoder----------多隐藏层版(Dense)

若为码字加上稀疏性约束。如果我们对隐层单元施加稀疏性约束的话，会得到更为紧凑的表达，
只有一小部分神经元会被激活。在Keras中，我们可以通过添加一个activity_regularizer达
到对某层激活值进行约束的目的：
encoded = Dense(encoding_dim, activation='relu',activity_regularizer=regularizers.activity_l1(10e-5))(encoder_input)

Created on Tue Nov 20 09:52:03 2018

@author: gear
"""

import numpy as np
from keras.datasets import mnist
from keras import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

np.random.seed(1)


def AutoEncoder(input_shape):
    '''
    实现一个3个隐层自编码器(AutoEncoder)
    Parameters:
        input_shape -- 输入的数据维度,如输入的一张28*28图片，shape=(28*28, )
    Returns:
        model -- 创建的AutoEncoder模型
    '''
    
    # 输入数据的shape
    encoder_input = Input(shape=(input_shape,))
    encoder_dim = 2
    
    # 建立编码层
    encoded = Dense(units=128, activation='relu')(encoder_input)
    encoded = Dense(units=64, activation='relu')(encoded)
    encoded = Dense(units=32, activation='relu')(encoded)
    encoder_output = Dense(units=encoder_dim)(encoded)
    
    # 建立解码层
    decoded = Dense(units=32, activation='relu')(encoder_output)
    decoded = Dense(units=64, activation='relu')(decoded)
    decoded = Dense(units=128, activation='relu')(decoded)
    decoded = Dense(units=784, activation='tanh')(decoded)
    
    # 建立自编码模型
    autoencoder = Model(inputs=encoder_input, outputs=decoded)
    
    # 建立编码器模型
    encoder = Model(inputs=encoder_input, outputs=encoder_output)
    
    return autoencoder, encoder

if __name__ == '__main__':
    # 载入数据        
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    # 数据预处理
    X_train = X_train.astype(np.float32)/255. - 0.5         # 将数据映射到-0.5 -- 0.5
    X_test = X_test.astype(np.float32)/255. - 0.5
    X_train = X_train.reshape(X_train.shape[0], -1)                       
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # 创建一个模型
    input_shape = X_train.shape[-1]
    autoencoder, encoder = AutoEncoder(input_shape)
    
    # 编译模型
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # 训练模型
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, 
                    validation_split=0.2, verbose=2)
    
    # 压缩成2维的mnist数据
    encoded_imgs = encoder.predict(X_test)      
    plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=Y_test, s=3)  
    plt.colorbar()  
    plt.show() 
    
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

 