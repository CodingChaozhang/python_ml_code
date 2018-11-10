# -*- coding: utf-8 -*-
"""
Note:
    实现卷积网络（CNN）中的关键算法

Created on Fri Nov  9 15:54:10 2018

@author: gear
"""

import numpy as np


def zero_pad(X, pad):
    '''
    图像填充，对输入的图像数据进行padding操作
    Parameters:
        X - 输入图像数据集 shape=(m, n_h, n_w, n_c)
        pad - 整数， 每个图像在height和width上的填充量
    Returns:
        X_pad - 填充后的图像数据集
    '''
    X_pad = np.pad(X, ((0,0), (pad, pad), (pad,pad), (0,0)), 'constant', constant_values=0)
    
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    '''
    对图像上的部分区域做一次卷积运算，并加上偏置
    Parameters:
        a_slice_prev - 输入图像数据对应卷积核大小的切片，shape=(f, f, n_c_prev)
        W - 卷积核(权重矩阵)， shape=(f, f, n_c_prev)
        b - 偏置， shape=(1,1,1)
    Returns:
        Z - 标量， 图像对应区域和卷积核做一次卷积后对应的数值
    '''
    s = np.multiply(a_slice_prev, W) + b
    Z = np.num(s)
    
    return Z

def conv_linear_forward(A_prev, W, b, hparameters):
    '''
    对图像数据集做线性向前传播卷积运算
    Parameters:
        A_prev - 前一层网络的输出值， shape=(m, n_h_prev, n_w_prev, n_c_prev)
        W - 卷积核张量，shape=(f, f, n_c_prev, n_c) ----------注意这里包含了n_c个卷积核！
        b - 偏置张量， shape=(1, 1, 1, n_c)
        hparameters - 包含超参数pad和stride的字典
    Returns:
        Z - 进行卷积后的图像数据集
        cache - 存储反向传播需要变量
    '''
    # step1 获取输入图像数据的shape
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    # step2 获取卷积核张量的shape
    (f, f, n_c_prev, n_c) = W.shape
    # step3 获取超参数信息
    pad = hparameters['pad']
    stride = hparameters['stride']
    # step4 对输入图像数据进行padding
    A_prev_pad = zero_pad(A_prev, pad)
    # step5 计算卷积操作后图像的高和宽
    n_h = np.floor((n_h_prev + 2*pad - f) / stride) + 1
    n_w = np.floor((n_w_prev + 2*pad - f) / stride) + 1
    # step6 初始化卷积后的图像数据
    Z = np.zeros((m, n_h, n_w, n_c))
    # step6 对输入图像数据进行卷积操作
    for i in range(m):                      #----------思考这里如何向量化？------------#
        a_prev_pad = A_prev_pad[i]          #获取单张图像, shape=(n_h_prev, n_w_prev, n_c_prev),这里应该为pad后的参数
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):        # n_c表示本层网络卷积核的个数
                    # 获取单次卷积的窗口滑动参数
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # 获取单次卷积的窗口
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]  # 注意这里对所有通道同时做切片
                    # 对每一个对于的窗口进行卷积运算
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[..., c], b[..., c])
    
    # 检查卷积后图像集的shape
    assert(Z.shape == (m, n_h, n_w, n_c))
    
    # 缓存反向传播需要的参数
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

def conv_activate_forward(Z, cache, activation='relu'):
    '''
    将卷积后的结果输入到激活函数中
    Parameters:
        Z - 进行卷积后的图像数据集，shape=(m, n_h, n_w, n_c)
        cache - 缓存反向传播需要的参数
        activation - 激活函数类型
    Returns:
        A - 激活函数输出的图像数据集，shape=(m, n_h, n_w, n_c)
        cache - 缓存
    '''
    # step1 获取图像数据集的shape
    (m, n_h, n_w, n_c) = Z.shape
    # step2 初始化激活后的图像数据
    A = np.zeros((m, n_h, n_w, n_c)) 
    # step3 将数据输入激活函数中
    for i in range(m):
        if activation == 'relu':
            A[i] = relu(Z[i])
        elif activation == 'sigmoid':
            A[i] = sigmoid(Z[i])
        elif activation == 'tanh':
            A[i] = tanh(Z[i])
        else:
            print('所选激活函数不存在')
            exit(1)
            
    assert(A.shape == (m, n_h, n_w, n_c))
    cache = (cache, A)
    
    return A, cache


def pooling_forward(A_prev, hparameters, mode='max' ):
    '''
    对进行卷积后的图像数据集进行池化(pooling)操作
    Parameters:
        A_prev - 卷积后的图像数据集 shape=(m, n_h_prev, n_w_prev, n_c_prev)
        haprameters - 包含f和stride的字典---------一般情况下池化没有超参数pad
        mode - 池化类型 max/average
    Returns:
        A - 池化后的图像数据集
        cache - 缓存的反向传播所需参数
    '''
    # step1 获取输入图像数据集的shape
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    # step2 获取超参数
    f = hparameters['f']
    stride = hparameters['stride']
    # step3 计算池化后的图像尺寸
    n_h = np.floor((n_h_prev - f) / stride) + 1
    n_w = np.floor((n_w_prev - f) / stride) + 1
    n_c = n_c_prev                              # 池化不改变通道数
    # step4 对池化后的图像数据初始化
    A = np.zeros((m, n_h, n_w, n_c))
    
    for i in range(m):                              #----------思考这里如何向量化？------------#
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    # 获取单次pooling滑动参数
                    vert_start = h * stride
                    vert_end = vert_start + f       # f 表示池化核大小
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # 获取单次pooling窗口
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]  # 注意这里对每个通道分别做pooling
                    # 计算对应窗口pooling后的值
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)
                    else:
                        print('所选pooling类型不存在')
                        exit(1)
                    
                    
    # 检查pooling后的shape是否正确
    assert(A.shape == (m, n_h, n_w, n_c))
    cache = (A_prev, hparameters)
    
    return A, cache

def conv_backward(dZ, cache):
    '''
    实现卷积层的反向传播
    Parameters:
        dZ - 卷积层输出Z的梯度，shape=(m, n_h, n_w, n_c)
        cache - 反向传播需要的参数，conv_linear_forward的输出
    Returns:
        dA_prev - 卷积层输入(A_prev)的梯度值，shape=(m, n_h_prev, n_w_prev, n_c_prev)
        dW - 卷积层卷积核的梯度，shape=(f, f, n_c_prev, n_c)
        db - 卷积层偏置的梯度， shape=(1, 1, 1, n_c)
    '''
    
    # step1 获取cache的值
    (A_prev, W, b, hparameters) = cache
    # step2 获取A_prev的shape
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    # step3 获取卷积核W的shape
    (f, f, n_c_prev, n_c) = W.shape
    # step4 获取hparmeters的值
    pad = hparameters['pad']
    stride = hparameters['stride']
    # step5 获取dZ的shape
    (m, n_h, n_w, n_c) = dZ.shape
    
    # step5 初始化dA_prev, dW, db
    dA_prev = np.zeros((m, n_h_prev, n_w_prev, n_c_prev))
    dW = np.zeros((m, f, f, n_c_prev, n_c))
    db = np.zeros((1,1,1,n_c))
    
    # step6 对A_prev和dA_prev进行padding操作-------------------?
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):      # 遍历所有训练图像数据集
        # 获取单个训练
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    # 定位切片位置
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride 
                    horiz_end = horiz_start + f
                    #获取切片
                    a_prev_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # 计算da_prev_pad--------类比全连接网络中的dA_pre = np.dot(W.T, dZ)
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[..., c] * dZ[i, h, w, c]
                    # 计算dW ----------------类比全连接网络中的dW = np.dot(dZ, A_pre.T) / m
                    dW[..., c] += dZ[i, h, w, c] * a_prev_slice
                    # 计算db ----------------类比全连接网络中的db = np.sum(dZ, axis=1, keepdims=True) / m
                    db[..., c] += dZ[i, h, w, c]
        # 计算dA_prev
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:pad, :]
        
    assert(dA_prev,shape == (m, n_h_prev, n_w_prev, n_c_prev))
    
    return dA_prev, dW, db

def creat_mask_from_window(x):
    '''
    从输入矩阵中创建mask,以保存最大值的矩阵的位置
    目的：记录池化层最大值的位置，才能进行反向传播
    Parameters:
        x - shape=(f, f)
    Returns:
        mask - 包含x的最大值的位置的matrix, shape=(f, f)
    '''
    
    mask = (x == np.max(x))
    
    return mask

def distribute_value(dz, shape):
    '''
    给定一个值，按矩阵大小平均分配到矩阵中的每个位置
    目的：当进行average_pooling时，每个对应位置的值都有贡献，且贡献值相同
    Parameters:
        dz - 输入的实数
        shape - tuple，值为(n_h, n_w)
    Returns:
        a - 分配好值的矩阵，shape=(n_h, n_w)
    '''
    # 获取矩阵大小
    (n_h, n_w) = shape
    # 计算平均值
    average = dz / (n_h * n_w)
    # 填充矩阵
    a = np.ones(shape) * average
    
    return a

def pooling_backward(dA, cache, mode='max'):
    '''
    对pooling层进行反向传播
    Parameters:
        dA - 池化层向前传播输出的梯度， 和A的梯度shape相同，shape=(m, n_h, n_w, n_c)
        cache - 池化层向前传播时存储的参数（f, stride)
        mode - 池化方法 max/average
    Returns:
        dA_prev - 池化层向前传播输入的梯度，和A_prev的shape相同，shape= (m, n_h_prev, n_w_prev, n_c_prev)
    '''
    # step1 获取cache中的值
    (A_prev, hparameters) = cache
    
    # step2 获取超参数f和stride
    f = hparameters['f']
    stride = hparameters['stride']
    
    # step3 获取A_prev 和 dA 的shape
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    (m, n_h, n_w, n_c) = dA.shape
    
    # step4 初始化dA_prev
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):      # 遍历所有训练图像数据集
        # 选取单个数据
        a_prev = A_prev[i]
        
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    # 定位切片位置
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # 计算梯度
                    if mode == 'max':
                        # 获取切片
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # 为切片创建mask
                        mask = creat_mask_from_window(a_prev_slice)
                        # 计算dA_prev
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                    elif mode == 'average':
                        # 获取dA的值
                        da = dA[i, h, w, c]
                        # 确定pooling核的shape
                        shape = (f, f)
                        # 计算dA_prev
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
                        
    
    # 检查dA_prev的shape
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev


                        
                    
    
        


    
        
                   
                    
                    
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
        
        
        
        
        

    