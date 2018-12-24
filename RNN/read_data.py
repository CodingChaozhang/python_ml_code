# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:04:06 2018

@author: dell
"""

import dataset_utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
from keras.optimizers import RMSprop
#

#fish_data = 'data/fish_data.csv'
#fish_data = pd.read_csv(fish_data, index_col=0)
#dataset = fish_data.values
#
#dataset = dataset[0:200]
#dataset = dataset[:, 0:49]
#
#dataset_X = dataset[:, 0:-1]
##dataset_X = dataset_X / dataset_X.max()
##dataset_X = np.abs(dataset_X)
#
#dataset_Y = dataset[:, -1].reshape(-1, 1)
#s = dataset_Y.max()
#dataset_Y = dataset_Y / s
#
#scaler1 = StandardScaler()
#scaler2 = StandardScaler()
#scaler3 = StandardScaler()
#
#scaler4 = MinMaxScaler()
#
##dataset_X = scaler4.fit_transform(dataset_X)
#
#Ux = dataset[:, 0:-1:3]
#Uy = dataset[:, 1:-1:3]
#W = dataset[:, 2:-1:3]
#
#Ux = scaler1.fit_transform(Ux)
#Uy = scaler2.fit_transform(Uy)
#W = scaler3.fit_transform(W)
#
#temp = np.column_stack((Ux, Uy))
#dataset_X = np.column_stack((temp, W))
#
#
#dataset1 = np.column_stack((dataset_X, dataset_Y))

fish_data = 'H:/job_2/py_code/filament_para25.csv'
fish_data = pd.read_csv(fish_data, index_col=0)      # 探测器在细丝上方的数据
dataset = fish_data.values

dataset_X = dataset[:, 0:-3]
dataset_Y = dataset[:, -3].reshape(-1, 1)
s = dataset_Y.max()
dataset_Y = dataset_Y / s

Ux = dataset_X[:, 0:-1:3]
Uy = dataset_X[:, 1:-1:3]
W  = dataset[:, 2:-1:3]

scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()

Ux = scaler1.fit_transform(Ux)
Uy = scaler2.fit_transform(Uy)
W = scaler3.fit_transform(W)

temp = np.column_stack((Ux, Uy))
#dataset_X = np.column_stack((temp, W))   
dataset_X = temp
dataset1 = np.column_stack((dataset_X, dataset_Y))


train_X, train_Y = dataset_utils.generator_muti(dataset1, lookback=6, delay=0, min_index=0, max_index=150, step=1, batch_size=100)
test_X, test_Y = dataset_utils.generator_muti(dataset1, lookback=6, delay=0, min_index=150, max_index=None, step=1, batch_size=100)



def RNN_Model(input_shape):
    '''
    2D流场时间序列预测网络结构
    Parameters:
        input_shape -- 输入数据类型
    Returns:
        model -- 用于预测的模型
    '''
    Input = keras.layers.Input(shape=input_shape)
    X = keras.layers.LSTM(units=128, return_sequences=True)(Input)
#    X = keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(Input)
#    X = keras.layers.MaxPool1D(pool_size=2)(X)
#    
    X = keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')(X)  
    X = keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu')(X)
    X = keras.layers.Conv1D(filters=8, kernel_size=1, activation='relu')(X)
    X = keras.layers.Conv1D(filters=4, kernel_size=1, activation='relu')(X)
    X = keras.layers.LSTM(units=128, return_sequences=True)(X)
#    X = keras.layers.LSTM(units=128, return_sequences=True)(X)
    X = keras.layers.LSTM(units=256)(X)
#    X = keras.layers.Dense(units=128, activation='relu')(X)
#    X = keras.layers.Dense(units=128, activation='relu')(X)
#    X = keras.layers.Dense(units=128, activation='relu')(X)
#    X = keras.layers.Bidirectional(keras.layers.LSTM(units=128))(X)
    Output = keras.layers.Dense(units=1, activation='relu')(X)
    
    model = keras.models.Model(inputs=Input, outputs=Output)
    
    return model

def ANN_model(input_shape):
    Input = keras.layers.Input(shape=input_shape)
    # 第01层网络
    X = keras.layers.Dense(units=192, activation='relu')(Input)
    X = keras.layers.Dropout(rate=1)(X)
    
    # 第02层网络
    X = keras.layers.Dense(units=192, activation='relu')(X)
    X = keras.layers.Dropout(rate=1)(X) 
    
    # 第03层网络
    X = keras.layers.Dense(units=256, activation='relu')(X)
    X = keras.layers.Dropout(rate=1)(X)
    
    # 第04层网络
    X = keras.layers.Dense(units=256, activation='relu')(X)
    X = keras.layers.Dropout(rate=1)(X)
    
    # 第05层网络
    X = keras.layers.Dense(units=192, activation='relu')(X)
    X = keras.layers.Dropout(rate=1)(X)
    
    # 第06层网络
    X = keras.layers.Dense(units=48, activation='relu')(X)
    X = keras.layers.Dropout(rate=1)(X)
    
    # 第07层网络
    X = keras.layers.Dense(units=48, activation='relu')(X)
    X = keras.layers.Dropout(rate=1)(X)
    
    Sx = keras.layers.Dense(units=1, name='Sx')(X)  #x方向距离
    
    model = keras.models.Model(inputs=Input, outputs=Sx)
    
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
    
    model.compile(optimizer=RMSprop(lr=0.001), loss='mae')
    history = model.fit(train_X, train_Y, epochs=500, batch_size=50, validation_data=(test_X, test_Y),
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
#    y_true = scaler_Y.inverse_transform(test_Y)
#    y_pred = scaler_Y.inverse_transform(pred_Y)
    
    y_true = test_Y 
    y_pred = pred_Y 
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.legend(['y_true', 'y_pred'])
    plt.show()
    # calcualte RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print('Test RMSE: %.3f' % rmse)
    
    return y_true, y_pred



##############################---Main_Code---#################################
    
#train_X = train_X.reshape(-1, 192)
#test_X = test_X.reshape(-1, 192)

input_shape = train_X.shape[1:]    

# step1 建立模型
model = RNN_Model(input_shape)
model.summary()
#
# step2 进行训练
model = Train_Model(model, train_X, train_Y, test_X, test_Y)

# step3 进行预测
y_true, y_pred = Prediction(model, test_X, test_Y)
