# -*- coding: utf-8 -*-
"""
简介：
二分类SVM---by sklearn
Created on Mon Dec 24 16:57:48 2018

@author: gear
"""
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt

def load_data():
    
    iris = datasets.load_iris()    
    X = iris.data[:, :2]
    Y = np.array([1 if label==0 else -1 for label in iris.target])
    
    index = [i for i in range(len(X))]
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]
    
    train_X = X[0:100]
    train_Y = Y[0:100].reshape(-1,1)
    
    test_X = X[100:]
    test_Y = Y[100:].reshape(-1,1)
    
    return train_X, train_Y ,test_X, test_Y



    
train_X, train_Y ,test_X, test_Y = load_data()

clf = SVC(C=20, kernel='rbf')
clf.fit(train_X, train_Y)
pred = clf.predict(test_X)
pred = pred.reshape(-1, 1)

acc = 1 - np.mean(np.abs(pred - test_Y) / 2)



