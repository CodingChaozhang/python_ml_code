# -*- coding: utf-8 -*-
"""
简介：
实现人脸识别经典算法FaceNet

Created on Fri Nov 16 14:59:31 2018

@author: dell
"""

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers import BatchNormalization, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Flatten, Lambda, Concatenate
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

K.set_image_data_format('channels_first')

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import genfromtxt


def triplet_loss(y_true, y_pred, alpha=0.2):
    '''
    实现FaceNet中的损失函数
    Parameters:
        y_true - true标签，
        y_pred - 列表，包含了以下参数：
                anchor - 给定的anchor图像的编码，shape=(None, 128)
                positive - positive图像的编码， shape=(None, 128)
                negative - negative图像的编码， shape=(None, 128)
         alpha - 超参数
    Returns:
        loss - 实数， 损失函数的值
    '''
    
    # step1 获取anchor, positive, negative的图像编码
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # step2 计算anchor与positive之间的编码距离
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)     # 注意这里按列求和
    
    # step3 计算anchor与negative之间的编码距离
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    
    # step4 计算 pos_dist - neg_dist再加上两者之间的间距alpha
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    
    # step5 通过取带零的最大值和训练样本的求和来计算整个公式
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    
    return loss


    
    
    
    
    
    
