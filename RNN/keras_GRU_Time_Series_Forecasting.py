# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:07:59 2018

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam

import dataset_utils

# 读入数据

root = 'data/'
dataframe = pd.read_csv(root+'international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)

dataset = dataframe.values
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0,1))  # 将数据归一化到0-1区间
dataset = scaler.fit_transform(dataset)

dataset2 = np.arange(0, 150).reshape(150, 1)

dataset2 = scaler.fit_transform(dataset2.astype('float32'))





train_X, train_Y = dataset_utils.generator(dataset, lookback=4, delay=0, min_index=0, max_index=100, step=1, batch_size=100)
test_X, test_Y = dataset_utils.generator(dataset, lookback=4, delay=0, min_index=100, max_index=None, step=1, batch_size=100)




def RNN_Model(input_shape):
    '''
    时间序列预测网络结构
    Parameters:
        input_shape -- 输入数据类型
    Returns:
        model -- 用于预测的模型
    '''
    Input = keras.layers.Input(shape=input_shape)
#    X = keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(Input)
#    X = keras.layers.GRU(units=64, dropout=0.1, recurrent_dropout=0.5, return_sequences=True)(X)
    X = keras.layers.GRU(units=128, return_sequences=True)(Input)
    X = keras.layers.GRU(units=128)(X)
    Output = keras.layers.Dense(units=1)(X)
    
    model = keras.models.Model(inputs=Input, outputs=Output)
    
    return model

def CNN_model(input_shape):
    '''
    卷积序列预测网络结构
    Parameters:
        input_shape -- 输入数据类型
    Returns:
        model -- 用于预测的模型
    '''
    Input = keras.layers.Input(shape=input_shape)
#    X = keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(Input)
#    X = keras.layers.GRU(units=64, dropout=0.1, recurrent_dropout=0.5, return_sequences=True)(X)
    X = keras.layers.GRU(units=128, return_sequences=True)(Input)
    X = keras.layers.GRU(units=128)(X)
    Output = keras.layers.Dense(units=1)(X)
    
    model = keras.models.Model(inputs=Input, outputs=Output)
    
    return model
    

def Train_Model(model, train_X, train_Y, test_X, test_Y):
    '''
    对模型进行训练
    Parameters:
        model -- 建立的模型
        train_X -- 训练数据
        train_Y -- 训练标签
    Returns:
        model -- 训练好的模型
    '''
    
    model.compile(optimizer=Adam(lr=0.001), loss='mae')
    history = model.fit(train_X, train_Y, epochs=500, batch_size=100, validation_data=(test_X, test_Y),
              verbose=2, shuffle=False)
    
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    
    return model

def Prediction(model, test_X, test_Y):
    '''
    对测试数据进行预测
    Parameters:
        model -- 训练好的模型
        test_X -- 测试数据
        test_Y --测试标签
    Returns:
        None
    '''
    
    pred_Y = model.predict(test_X)
    
    # invert data to before:
    test_Y = test_Y.reshape(-1, 1)
    y_true = scaler.inverse_transform(test_Y)
    y_pred = scaler.inverse_transform(pred_Y)
    
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.legend(['y_true', 'y_pred'])
    plt.show()
    # calcualte RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print('Test RMSE: %.3f' % rmse)
    
    return y_true, y_pred



##############################---Main_Code---#################################

input_shape = train_X.shape[1:]    

# step1 建立模型
model = RNN_Model(input_shape)
#
## step2 进行训练
#model = Train_Model(model, train_X, train_Y, test_X, test_Y)
#
## step3 进行预测
#y_true, y_pred = Prediction(model, test_X, test_Y)





