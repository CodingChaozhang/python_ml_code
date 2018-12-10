# -*- coding: utf-8 -*-
"""
简介：
keras word Embedding 

Created on Mon Dec 10 16:25:14 2018

@author: dell
"""
import numpy as np
from keras.datasets import imdb
from keras import preprocessing 

max_features = 10000            # 作为特征的单词的个数
maxlen = 100                     # 在这么多单词之后截断文本

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=max_features)


# 将整数list转化为形状为(samples, maxlen)的二维张量
X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)  
X_test =  preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

