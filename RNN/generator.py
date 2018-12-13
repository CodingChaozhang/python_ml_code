# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:41:16 2018

@author: dell
"""

import numpy as np

import keras
from keras import layers
from keras.models import Sequential
import matplotlib.pyplot as plt
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


data = np.arange(0, 100).reshape(100, 1)

train_X, train_Y = generator(data, lookback=2, delay=0, min_index=0, max_index=80, step=1, batch_size=1000)
test_X, test_Y = generator(data, lookback=2, delay=0, min_index=80, max_index=None, step=1, batch_size=1000)
#train_X = train_X / 100
#train_Y = train_Y / 100
#
#test_X /= 100
#test_Y /= 100

batch_size = 50

#max_len = 5
#train_X2 = pad_sequences(train_X, maxlen=max_len, dtype='float32')



def model1():
    '''
    这里需要epochs=2000效果才好
    '''
    model = Sequential()
#    model.add(layers.GRU(units=32, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(layers.GRU(units=32, input_shape=(None, 1)))    
    model.add(layers.Dense(units=1))
    model.compile(optimizer='Adam', loss='mae')
    model.fit(train_X, train_Y, epochs=2000, batch_size=100, shuffle=False)
    
    
    pred = model.predict(test_X)
    pred *= 100
    
    plt.plot(pred)
    plt.plot(test_Y*100)
    
    for i in range(len(test_Y)):
        print(test_X[i]*100, '-->', np.round(pred[i]))
        

def model2():
    '''
    采用stateful GRU的特点是，在处理过一个batch的训练数据后，其内部状态（记忆）
    会被作为下一个batch的训练数据的初始状态。状态GRU使得我们可以在合理的计算复杂度内处理较长序列。
    '''
    batch_size = 1
    model = Sequential()
    model.add(layers.GRU(units=32, batch_input_shape=(batch_size, train_X.shape[1], train_X.shape[2]), stateful=True))
    model.add(layers.Dense(units=1))
    model.compile(optimizer='Adam', loss='mae')
    for i in range(300):
        model.fit(train_X, train_Y, epochs=1, batch_size=batch_size, shuffle=False)
        model.reset_states()
        
    preds = []
    for i in range(len(test_Y)):        
        pred = model.predict(test_X[i].reshape(1, 2, 1))
        pred *= 100
        preds.append(np.round(pred.flatten().tolist()))
        
    plt.plot(preds)
    plt.plot(test_Y*100)
    
    for i in range(len(test_Y)):
        print(test_X[i]*100, '-->', np.round(preds[i].flatten().tolist()))

    return model

def model3():
    # 可以预测从长度1-5的任意序列
    num_inputs = 1000
    max_len = 5
    dataX = []
    dataY = []
    for i in range(num_inputs):
        start = np.random.randint(len(data)-2)
        end = np.random.randint(start, min(start+max_len,len(data)-1))
        sequence_in =data[start:end+1]
        sequence_out = data[end + 1]
        
        dataX.append(sequence_in)
        dataY.append(sequence_out)
        
    
    X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
    # reshape X to be [samples, time steps, features]
    train_X3 = np.reshape(X, (X.shape[0], max_len, 1))
    # normalize
    
    train_X3 /= 100
    train_Y3 = np.array(dataY)
    
    
    
    batch_size = 1
    model = Sequential()
    model.add(layers.GRU(32, input_shape=(train_X3.shape[1], 1)))
    model.add(layers.Dense(units=1))
    model.compile(optimizer='Adam', loss='mae')
    model.fit(train_X3, train_Y3, epochs=50, batch_size=batch_size, verbose=2)
    
    preds = []
    test_X3 = []
    for i in range(20):
        pattern_index = np.random.randint(len(dataX))
        pattern = dataX[pattern_index]
        x = pad_sequences([pattern], maxlen=max_len, dtype='int32')
        x = np.reshape(x, (1, max_len, 1))
        x = x / 100
        pred = model.predict(x)
        preds.append(pred)
        test_X3.append(x)
    
    for i in range(len(test_Y)):
        print(test_X[i]*100, '-->', np.round(preds[i]))
        
    
###############################################################################

model1()    


    






