# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:14:35 2018

@author: GEAR
"""

'''
使用sklearn进行knn分类
'''
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 读取数据
dir = 'D:/machine learing/python-ml-code/datingTestSet.txt'
df = pd.read_table(dir, header=None)
df['class'] = df.iloc[:,-1].astype('category')
# 1代表didntLike, 2代表smallDoses, 3代表largeDoses
df['class'].cat.categories = [1, 3, 2]
dataSet = df.iloc[:, 0:-2].values
labels = df.iloc[:,-1].tolist()

#归一化
scaler = MinMaxScaler()
dataSet = scaler.fit_transform(dataSet)

# 交叉分类
train_X,test_X, train_y, test_y = train_test_split(dataSet,
                                                   labels,
                                                   test_size=0.2) # test_size:测试集比例20%

# KNN模型，选择3个邻居
model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_X, train_y)
print(model)

expected = test_y
predicted = model.predict(test_X)
print(metrics.classification_report(expected, predicted))       # 输出分类信息
label = list(set(labels))    # 去重复，得到标签类别
print(metrics.confusion_matrix(expected, predicted, labels=label))  # 输出混淆矩阵信息
