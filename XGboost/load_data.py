# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 20:20:31 2018

@author: gear
github:  https://github.com/gear106
"""

import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

##################################--parameters--###############################

r = 0.1     # 细丝刚度                             
y = 0.0     # 竖直距离
v = 0.7     # 巡航速度
xt = 9      #读数据圆心的位置

###############################################################################
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

###############################################################################
root = 'H:/job_2/'
root2 = 'data/'
path_test = 'H:/job_2/vortex/temp25_test2/'

dataset_X, dataset_Y = Input_X(root, path_test, y, v, r)  
dataset_X = dataset_X.reshape(200,-1) 
dataset_Y = dataset_Y + 1.5
dataset = np.column_stack((dataset_X, dataset_Y))

temp = pd.read_csv(root2+'train.csv')
temp = temp.iloc[:,1:1877]

temp1 = pd.read_csv(root2+'test.csv')
temp1 = temp1.iloc[:,1:1877]

temp = np.row_stack((temp, temp1))
temp = np.row_stack((temp, dataset))
pd = pd.DataFrame(data=temp)

pd.to_csv(root2 + 'train1.csv')


