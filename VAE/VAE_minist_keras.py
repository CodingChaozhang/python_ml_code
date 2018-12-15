# -*- coding: utf-8 -*-
"""
简介：
VAE编码网络keras实现
Created on Fri Dec 14 19:59:18 2018

@author: dell
"""

import keras 
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np
from keras.datasets import mnist
from scipy.stats import norm
import matplotlib.pyplot as plt

img_shape =(28, 28, 1)
batch_size = 16
latent_dim =2           # 潜在空间的维度(2维)

###################################encoder######################################
input_img = layers.Input(shape=img_shape)

X = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(input_img)
X = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', strides=(2, 2))(X)
X = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(X)
X = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(X)

shape_before_flattening = K.int_shape(X)        #以整数Tuple或None的形式返回张量shape
X = layers.Flatten()(X)
X = layers.Dense(units=32, activation='relu')(X)

# 输入图像最终被编码为两个参数：
z_mean = layers.Dense(units=latent_dim)(X)      #shape=(?, 2)
z_log_var = layers.Dense(units=latent_dim)(X)


###############################sampling#########################################
def sampling(args):
    '''潜在空间的采样函数'''
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), 
                              mean=0, stddev=1.0)    #shape=(?, 2)
    
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])     #shape=(?, 2)


###############################decoder##########################################
decoder_input = layers.Input(K.int_shape(z)[1:])    #shape=(?, 2)

# 对输入进行上采样
X = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)    #np.prod() 函数计算数组元素乘积等
# 将z转换为特征图, 使其形状与编码器模型最后一个Flatten层之前的特征图的形状相同
X = layers.Reshape(shape_before_flattening[1:])(X)  #shape=(?, 28, 28, 64)

# 使用反卷积和卷积层将z解码为和原始输入图像具有相同尺寸的特征图
X = layers.Conv2DTranspose(filters=32, kernel_size=3, padding='same', activation='relu', strides=(2,2))(X) #shape=(?, ?, ?, 32)
X = layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(X)  #shape=(?, ?, ?, 1)

decoder = Model(inputs=decoder_input, outputs=X)

# 得到解码的z
z_decoded = decoder(z)

# 定义用于计算VAE的损失函数
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
        return K.mean(xent_loss + kl_loss)
    
    # 通过编写call方式来实现一个自定义层
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        # 我们不使用这个输出， 但层必须要有返回值
        return x

# 对输入和解码后的输出调用自定义层，得到最终的模型输出
y = CustomVariationalLayer()([input_img, z_decoded])


##################################Training-model###############################

# 载入训练数据
(train_X, _), (test_X, test_Y) = mnist.load_data()

train_X = train_X.astype('float32') / 255
train_X = train_X.reshape(train_X.shape + (1,))
test_X = test_X.astype('float32') / 255
test_X = test_X.reshape(test_X.shape + (1,))


# 训练模型
VAE = Model(inputs=input_img, outputs=y)
VAE.summary()
VAE.compile(optimizer='adam', loss=None)
VAE.fit(x=train_X, y=None, shuffle=True, epochs=10, batch_size=batch_size, validation_data=(test_X, None))

VAE.save('VAE.h5')

###############################Sampling########################################
# Display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# Linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z,
# since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

    