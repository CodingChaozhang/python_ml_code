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
import fr_utils
from inception_blocks_v2 import *

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


#创建模型
FaceNet = faceRecoModel(input_shape=(3, 96, 96))

#开始时间
start_time = time.clock()

#编译模型
FaceNet.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

#加载权值
fr_utils.load_weights_from_FaceNet(FaceNet)

#结束时间
end_time = time.clock()

#计算时差
minium = end_time - start_time

print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")

    
database = {}
database["danielle"] = fr_utils.img_to_encoding("images/danielle.png", FaceNet)
database["younes"] = fr_utils.img_to_encoding("images/younes.jpg", FaceNet)
database["tian"] = fr_utils.img_to_encoding("images/tian.jpg", FaceNet)
database["andrew"] = fr_utils.img_to_encoding("images/andrew.jpg", FaceNet)
database["kian"] = fr_utils.img_to_encoding("images/kian.jpg", FaceNet)
database["dan"] = fr_utils.img_to_encoding("images/dan.jpg", FaceNet)
database["sebastiano"] = fr_utils.img_to_encoding("images/sebastiano.jpg", FaceNet)
database["bertrand"] = fr_utils.img_to_encoding("images/bertrand.jpg", FaceNet)
database["kevin"] = fr_utils.img_to_encoding("images/kevin.jpg", FaceNet)
database["felix"] = fr_utils.img_to_encoding("images/felix.jpg", FaceNet)
database["benoit"] = fr_utils.img_to_encoding("images/benoit.jpg", FaceNet)
database["arnaud"] = fr_utils.img_to_encoding("images/arnaud.jpg", FaceNet)

def verify(image_path, identity, database, model):
    """
    对“identity”与“image_path”的编码进行验证。

    参数：
        image_path -- 摄像头的图片。
        identity -- 字符类型，想要验证的人的名字。
        database -- 字典类型，包含了成员的名字信息与对应的编码。
        model -- 在Keras的模型的实例。

    返回：
        dist -- 摄像头的图片与数据库中的图片的编码的差距。
        is_open_door -- boolean,是否该开门。
    """
    #第一步：计算图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = fr_utils.img_to_encoding(image_path, model)

    #第二步：计算与数据库中保存的编码的差距
    dist = np.linalg.norm(encoding - database[identity])

    #第三步：判断是否打开门
    if dist < 0.7:
        print("欢迎 " + str(identity) + "回家！")
        is_door_open = True
    else:
        print("经验证，您与" + str(identity) + "不符！")
        is_door_open = False

    return dist, is_door_open
 
    
verify("images/camera_0.jpg","younes",database, FaceNet)    
    
    
