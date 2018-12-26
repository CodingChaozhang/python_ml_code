# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:04:06 2018

@author: dell
"""
import os
import dataset_utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
from keras.optimizers import RMSprop

from keras.optimizers import Adagrad

root = 'H:/job_2/'
path1 = 'H:/job_2/vortex/temp25_1/'
path2 = 'H:/job_2/vortex/temp25_2/'
path3 = 'H:/job_2/vortex/temp25_3/'
path4 = 'H:/job_2/vortex/temp25_4/'
path5 = 'H:/job_2/vortex/temp25_5/'
path_test = 'H:/job_2/vortex/temp25_test/'

r = 0.1     # 细丝刚度                             
y = 0.0     # 竖直距离
v = 0.7     # 巡航速度
xt = 9      #读数据圆心的位置

def Distance(root, r, xt):
    '''
    计算不同刚度的细丝在不同时刻距离探测器的水平位置
    Parameters:
        root -- 文件根目录
        r -- 细丝刚度
    Returns:
        S -- 细丝距离探测器的水平位置
    '''
    path = root + 'data/r=' + str(r) + '/lag/'
    filenames=os.listdir(path)  #返回指定目录下的所有文件和目录名
    numbs = len(filenames)
    
    S = []      # 细丝头部距离探测器圆心的位置
    for i in range(numbs):
        with open(path + filenames[i]) as file:
            x = file.readline().strip().split()
            S.append(xt-float(x[0]))         # 这里9指探测器圆心水平位置
    
    return S

def Input_X(root, path, y, v, r):
    '''
    读取训练数据并做预处理
    Parameters:
        path -- 训练数据所在文件路径
        y -- 探测器相对鱼的Y距离
        v -- 游动细丝的巡航速度    
        r -- 游动细丝的刚度
    Returns:
        train_X -- Array, 每行数据为16个探测点的（Ux, Uy, W), shape=(m, 48)
        train_Y -- Array, 训练标签， 每行为(S, V, R), shape=(m, 3)
        test_X -- 同train_X
        test_Y -- 同trian_Y
    '''
    t = 25
    filenames=os.listdir(path)  #返回指定目录下的所有文件和目录名
    
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    scaler3 = StandardScaler()
    
#    Ux = scaler1.fit_transform(Ux)
#    Uy = scaler2.fit_transform(Uy)
#    W = scaler3.fit_transform(W)
        
    X = []
    numbs = len(filenames)
    for i in range(0, numbs):
        path1 = path + filenames[i] 
        df = pd.read_table(path1, header=None, skiprows=[0,1,2,3,4,5,6], sep='\s+')
        df.columns = ['X', 'Y', 'Ux', 'Uy', 'W']
        data = df.drop(['X','Y'], axis=1)
        temp = data.values
        Ux = scaler1.fit_transform(temp[:,0].reshape(-1,1))
        Uy = scaler2.fit_transform(temp[:,1].reshape(-1,1))
        W = scaler3.fit_transform(temp[:,2].reshape(-1,1))
        temp[:,0] = Ux.reshape(-1)
        temp[:,1] = Uy.reshape(-1)
        temp[:,2] = W.reshape(-1)
        
        temp = temp.reshape(t, t, 3)
        X.append(temp)
    X = np.array(X)
    Ux = scaler1.fit_transform(X[:,:,:,0].reshape(-1,1))
    Uy = scaler2.fit_transform(X[:,:,:,1].reshape(-1,1))
    W = scaler3.fit_transform(X[:,:,:,2].reshape(-1,1))
    X[:,:,:,0] = Ux.reshape(200,t,t)
    X[:,:,:,1] = Uy.reshape(200,t,t)
    X[:,:,:,2] = W.reshape(200,t,t)
    S = Distance(root, r, xt)                # 探测器相对鱼的X距离
    S = np.array(S)
                    
    return X, S


def RNN_Model(input_shape):
    '''
    2D流场时间序列预测网络结构
    Parameters:
        input_shape -- 输入数据类型
    Returns:
        model -- 用于预测的模型
    '''
    Input = keras.layers.Input(shape=input_shape)
#    X = keras.layers.LSTM(units=128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(Input)
    X = keras.layers.LSTM(units=256, return_sequences=True)(Input)

#    X = keras.layers.LSTM(units=128, return_sequences=True)(X)
#    X = keras.layers.LSTM(units=128, return_sequences=True)(X)
#    X = keras.layers.LSTM(units=128, return_sequences=True)(X)
#    X = keras.layers.LSTM(units=256, return_sequences=True)(X)
#    X = keras.layers.LSTM(units=256, return_sequences=True)(X)
    X = keras.layers.LSTM(units=256)(X)
#    X = keras.layers.Bidirectional(keras.layers.LSTM(units=128))(X)
#    X = keras.layers.Dense(units=128, activation='relu')(X)
#    X = keras.layers.Dense(units=128, activation='relu')(X)
#    X = keras.layers.Dropout(rate=0.2)(X)
#    X = keras.layers.Dense(units=128, activation='relu')(X)
    Output = keras.layers.Dense(units=1)(X)
    
    model = keras.models.Model(inputs=Input, outputs=Output)
    
    return model

def RNN_Model2(input_shape):
    '''
    2D流场时间序列预测网络结构
    Parameters:
        input_shape -- 输入数据类型
    Returns:
        model -- 用于预测的模型
    '''
    Input = keras.layers.Input(shape=input_shape)
    
    
    X = keras.layers.LSTM(units=512, return_sequences=True)(Input)
    
#    X = keras.layers.LSTM(units=256, dropout=0.2, recurrent_dropout=0.2, 
#                          kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(X)
#    X = keras.layers.LSTM(units=256, dropout=0.2, recurrent_dropout=0.2, 
#                          kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(X)
    
    
    X = keras.layers.LSTM(units=512, return_sequences=True)(X)  

    X = keras.layers.LSTM(units=512)(X)  
    X = keras.layers.Dense(units=512, activation='relu')(X)  
    Output = keras.layers.Dense(units=1)(X)
    
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
    history = model.fit(train_X, train_Y, epochs=200, batch_size=40, validation_data=(test_X, test_Y),
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
dataset_X, dataset_Y = Input_X(root, path1, y, v, r)  
dataset_X = dataset_X.reshape(200,-1) 
dataset1 = np.column_stack((dataset_X, dataset_Y))

dataset_X, dataset_Y = Input_X(root, path2, y, v, r)  
dataset_X = dataset_X.reshape(200,-1) 
dataset_Y = dataset_Y + 0.2
dataset2 = np.column_stack((dataset_X, dataset_Y))

dataset_X, dataset_Y = Input_X(root, path3, y, v, r)  
dataset_X = dataset_X.reshape(200,-1) 
dataset_Y = dataset_Y + 0.4
dataset3 = np.column_stack((dataset_X, dataset_Y))

dataset_X, dataset_Y = Input_X(root, path4, y, v, r)  
dataset_X = dataset_X.reshape(200,-1) 
dataset_Y = dataset_Y + 0.6
dataset4 = np.column_stack((dataset_X, dataset_Y))

dataset_X, dataset_Y = Input_X(root, path5, y, v, r)  
dataset_X = dataset_X.reshape(200,-1) 
dataset_Y = dataset_Y + 0.8
dataset5 = np.column_stack((dataset_X, dataset_Y))

dataset_X, dataset_Y = Input_X(root, path_test, y, v, r)  
dataset_X = dataset_X.reshape(200,-1) 
dataset_Y = dataset_Y + 0.5
dataset6 = np.column_stack((dataset_X, dataset_Y))


train_X1, train_Y1 = dataset_utils.generator_muti(dataset1, lookback=6, delay=0, min_index=0, max_index=None, step=1, batch_size=200)
train_X2, train_Y2 = dataset_utils.generator_muti(dataset2, lookback=6, delay=0, min_index=0, max_index=None, step=1, batch_size=200)
train_X3, train_Y3 = dataset_utils.generator_muti(dataset3, lookback=6, delay=0, min_index=0, max_index=None, step=1, batch_size=200)
train_X4, train_Y4 = dataset_utils.generator_muti(dataset4, lookback=6, delay=0, min_index=0, max_index=None, step=1, batch_size=200)
train_X5, train_Y5 = dataset_utils.generator_muti(dataset5, lookback=6, delay=0, min_index=0, max_index=None, step=1, batch_size=200)

train_X = np.row_stack((train_X1, train_X2))
train_X = np.row_stack((train_X, train_X3))
train_X = np.row_stack((train_X, train_X4))
train_X = np.row_stack((train_X, train_X5))

train_Y = np.row_stack((train_Y1, train_Y2))
train_Y = np.row_stack((train_Y, train_Y3))
train_Y = np.row_stack((train_Y, train_Y4))
train_Y = np.row_stack((train_Y, train_Y5))


test_X, test_Y = dataset_utils.generator_muti(dataset6, lookback=6, delay=0, min_index=0, max_index=None, step=1, batch_size=200)



input_shape = train_X.shape[1:]    

# step1 建立模型
model = RNN_Model2(input_shape)
model.summary()
#
# step2 进行训练
model = Train_Model(model, train_X, train_Y, test_X, test_Y)

# step3 进行预测
y_true, y_pred = Prediction(model, test_X, test_Y)
