# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 21:25:31 2018

@author: GEAR
"""
import numpy as np
import pandas as pd
import collections
from sklearn import preprocessing 

def dataload1(dir):
    '''
    使用常规方法读并处理txt文件
    '''
    fr = open(dir, 'r')
    # 读取所有行
    lines = fr.readlines()
    # 得到文件行数
    numLines = len(lines)
    # 初始化数据矩阵
    features = np.zeros((numLines, 3)) #这里3表示输入数据的维数
    # 初始化标签向量
    labels = []
    
    row = 0
    for line in lines:
        line = line.strip().split('\t')
        features[row,:] = line[0:3] #将line的前3行取出来
        #根据文本中标记分类
        if line[-1] == 'didntLike':
            labels.append(1)
        elif line[-1] == 'smallDoses':
            labels.append(2)
        elif line[-1] == 'largeDoses':
            labels.append(3)
            
        row += 1
    return features, labels

def dataload2(dir):
    '''
    使用dataFrame处理txt文件
    '''
    df = pd.read_table(dir, header=None)
    df['class'] = df.iloc[:,-1].astype('category')
    # 1代表didntLike, 2代表smallDoses, 3代表largeDoses
    df['class'].cat.categories = [1, 3, 2]
    features = df.iloc[:, 0:-2].values
    labels = df.iloc[:,-1].tolist()
    
    return features, labels

def autoNorm1(dataSet):
    '''
    对数据进行归一化,输入数据为np.array，
    归一化公式：new = (old - min) /(max - min), 将数据归一化到0-1之间
    '''
    minVals = dataSet.min(axis=0) #统计每一个数据的最小值
    maxVals = dataSet.max(axis=0) 
    ranges = maxVals - minVals
    
    normDataSet = np.zeros(np.shape(dataSet)) #归一化矩阵初始化
    rows = dataSet.shape[0] # 数据的行数
    normDataSet = dataSet - np.tile(minVals, (rows, 1)) # 原始数据减去最小值
    normDataSet = normDataSet / np.tile(ranges, (rows, 1))
    
    return normDataSet
    
def autoNorm2(dataSet):
    '''
    采用sklearn对数据进行归一化
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    normDataSet = min_max_scaler.fit_transform(dataSet)
    
    return normDataSet


def KNeighbors(testData, trainData, trainLabels, k):
    '''
    testData:    测试数据
    trainData:   训练数据
    trainLabels: 训练标签
    k          : 选择距离最小的k个点
    '''
    rows = trainData.shape[0] # 测试数据行数
    #计算欧式距离
    dist = np.sum((testData - trainData)**2, axis=1)**0.5
    # k个最近的标签
    k_labels = [trainLabels[index] for index in dist.argsort()[0:k]]
    # 出现次数最多的标签为最终类
    testlabel = collections.Counter(k_labels).most_common(1)[0][0]
    return testlabel

def KNeighborsClassifier(dataSet, labels):
    '''
    这里函数参数都是整理好的归一化后的参数
    '''
    # 获的数据集的行数
    rows = dataSet.shape[0]
    # 取所有数据的20%作为测试数据
    numTest = int(rows*0.2)
    #分类正确率
    correctCount = 0.0
    
    for i in range(numTest):
        #前numtest个数据作为测试集，后numtest个数据作为训练集
        classifierResult =  KNeighbors(dataSet[i], dataSet[numTest:], 
                                       labels[numTest:], 5)
        if classifierResult == labels[i]:
            correctCount += 1
    print('正确率：%f%%'%((correctCount/numTest)*100))
    
    
        
if __name__ == '__main__':
    dir = 'D:/machine learing/python-ml-code/datingTestSet.txt'
    # 导入数据
    dataSet, labels = dataload1(dir)
    # 数据归一化
    normDataSet = autoNorm1(dataSet)
    # 分类测试
    KNeighborsClassifier(normDataSet, labels)
    
        