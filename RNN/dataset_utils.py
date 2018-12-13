# -*- coding: utf-8 -*-
"""
简介：
生成时间序列预测的序列数据

Created on Tue Dec 11 10:58:58 2018

@author: dell
"""
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    '''
    用于生成训练RNN等网络的时间序列的函数
    parameters:
        data -- 原始的时间序列数据‘
        lookback -- 输入数据所包含过去时间步数
        delay -- 目标在未来多少个时间步之后
        min_index -- 数组中索引, 用于界定需要抽取那些时间步
        max_index -- 同上
        shuffle -- 是否打乱数据集
        batch_size -- 每个批量的样本数
        step -- 数据采集周期, 即每隔多少个点采样一次
    Yields:
        samples -- 输入数据的一个批量
        targets -- 对应的目标数组
    '''
        
    if max_index is None:
        max_index = len(data) - delay - 1  # 采样数据的最大时间步数
    i = min_index + lookback
    
    if shuffle:
        rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
    else:
        if i + batch_size >= max_index:
            i = min_index + lookback
        rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)
        
    samples = np.zeros((len(rows), lookback//step, data.shape[-1]))
    targets = np.zeros((len(rows), ))
    
    for j, row in enumerate(rows):
        indices = range(rows[j] - lookback, rows[j], step)
        samples[j] = data[indices]
        targets[j] = data[rows[j] + delay]
            
    return samples, targets

def generator2(dataset, max_len, step, max_index, delay):
    '''
    用于生产最大长度为n的任意长度时间序列数据
    Parameters:
        dataset -- 用于生成时间序列的数据集
        max_len -- 最长时间序列
        step -- 数据采样周期，即每隔多少个点采样
        delay -- 目标在未来多少个时间步之后
    Returns:
        samples -- 采样数据
        target -- 对应目标数据
    '''
    
    if max_index is None:
        max_index = len(data) - delay - 1  # 采样数据的最大时间步数
    num_inputs = 1000       # 数据样本数
    dataX = []
    dataY = []
    for i in range(num_inputs):
        start = np.random.randint(max_index)
        end = np.random.randint(start, min((start+max_len)*step, max_index))
        sequence_in =dataset[start: end+1 :step]
        sequence_out = dataset[end + step - 1 + delay]
        
        dataX.append(sequence_in)
        dataY.append(sequence_out)
        
    
    X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
    # reshape X to be [samples, time steps, features]
    train_X = np.reshape(X, (X.shape[0], max_len, 1))
    
    train_Y = np.array(dataY)
    
    return train_X, train_Y

def generator_muti(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    '''
    用于生成训练RNN等网络的时间序列的函数 -----多变量数据
    parameters:
        data -- 原始的时间序列数据‘
        lookback -- 输入数据所包含过去时间步数
        delay -- 目标在未来多少个时间步之后
        min_index -- 数组中索引, 用于界定需要抽取那些时间步
        max_index -- 同上
        shuffle -- 是否打乱数据集
        batch_size -- 每个批量的样本数
        step -- 数据采集周期, 即每隔多少个点采样一次
    Yields:
        samples -- 输入数据的一个批量
        targets -- 对应的目标数组
    '''
        
    if max_index is None:
        max_index = len(data) - delay - 1  # 采样数据的最大时间步数
    i = min_index + lookback

    
    if shuffle:
        rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
    else:
        if i + batch_size >= max_index:
            i = min_index + lookback
        rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)
        
    samples = np.zeros((len(rows), lookback//step, data.shape[-1]))
    targets = np.zeros((len(rows), 1))
    
    for j, row in enumerate(rows):
        indices = range(rows[j] - lookback, rows[j], step)
        samples[j] = data[indices]
        targets[j] = data[rows[j] + delay][-1]
    
    samples = samples[:, :, 0:-1]        
    return samples, targets

####################################test-code###################################
data = np.arange(0, 100).reshape(100, 1)

train_X, train_Y = generator2(data, max_len=5, step=2, max_index=None, delay=1)

#train_X, train_Y = generator(data, lookback=2, delay=0, min_index=0, max_index=80, step=1, batch_size=1000)
#test_X, test_Y = generator(data, lookback=2, delay=0, min_index=80, max_index=None, step=1, batch_size=1000)


