# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 21:24:18 2019

@author: gear
github:  https://github.com/gear106
"""

import numpy as np

X = np.random.random((10, 8))

def norm_X(X):
    '''
    对数据进行归一化处理
    '''
    mean = np.mean(X, axis=0)
    var = np.std(X, axis=0, ddof=1)
    norm_X = (X - mean) / var
    
    return norm_X

def SVD(X):
    '''
    对特征做SVD分解
    '''
    sigma = (X.T @ X) / X.shape[0]
    U, S, V = np.linalg.svd(sigma)
    
    return U, S, V

def decomp_X(X, U, k):
    '''
    压缩特征, k表示压缩的维数
    '''
    Z = X @ U[:, :k]
    
    return Z

def recover_X(Z, U, k):
    '''
    恢复原始数据维数
    '''
    X = Z @ U[:,:k].T
    
    return X

norm_X = norm_X(X) 
U, S, V = SVD(X)
decomp_X = decomp_X(X, U, 5)
recover_X = recover_X(decomp_X, U, 5)