# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:57:10 2018


@author: GEAR
"""
'''
Note:
     m -  样本个数
     f - 卷积核大小
    n_h - 图像高度
    n_w - 图像宽度
    n_c - 图像通道数
    pad - 图像填充数量
'''
import numpy as np
import matplotlib.pyplot as plt


def zero_pad(X, pad):
    '''
    对输入的图像数据进行padding填充
    Parameters:
        X - 图像数据集， 维度(m, n_h, n_w, n_c)
        pad - 整数， 每个图像在垂直和水平维度上的填充数
    Returns:
        X_pad - 填充后数据集, 维度(m, n_h+2*pad, n_w+2*pad, n_c)
    '''
    X_pad = np.pad(X, ((0,0), (pad, pad), (pad,pad), (0,0)), 'constant')
    
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    '''
    在前一层的激活输出的一个片段上和一个filter核W卷积
    Parameters:
        a_slice_prev - 单次卷积操作的部分图像区域，shape=(f, f, n_c_pre)
        W - 卷积核， shape=(f, f, n_c_pre)
        b - 偏置，包含在一个矩阵中，shape=(1,1,1)
    Returns:
        Z - 单次卷积操作后的值
    '''
    s = np.multiply(W, a_slice_prev) + b
    Z = np.sum(s)
    
    return Z

def conv_forward(A_pre, W, b, hparameters):
    '''
    对某一层的图像数据进行卷积运算
    Parameters:
        A_pre - 前一层网络的输出值，shape=(m, n_h_pre, n_w_pre, n_c_pre)
        W - 当前层的卷积核，shape=(f, f, n_c_pre, n_c)
        b - 当前层的偏置， shape=(1,1,1,n_c)
        hparameters - 包含超参数pad与stride的字典
    Returns:
        Z - 卷积后的输出，shape(m, n_h, n_w, n_c)
        cache - 缓存了一些反向传播需要的参数
    '''
    #step1 获取上一层数据的基本信息
    (m, n_h_pre, n_w_pre, n_c_pre) = A_pre.shape
    #step2 获取卷积核的基本信息
    (f, f, n_c_pre, n_c) = W.shape
    #step3 获取超参数pad和stride
    pad = hparameters['pad']
    stride = hparameters['stride']
    #step4 计算卷积核的图像尺寸
    n_h = np.floor((n_h_pre + 2*pad - f) / stride) + 1
    n_w = np.floor((n_w_pre + 2*pad - f) / stride) + 1
    #step5 初始化卷积核的图像数据
    Z = np.zeros(m, n_h, n_w, n_c)
    #step6 对图像进行padding操作
    A_pre_pad = zero_pad(A_pre, pad)
    #step7 对数据集删的所有图像进行卷积运算
    for i in range(m):
        a_pre_pad = A_pre_pad[i]    #单个样本
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    #滑动切片
                    a_slice_pre = a_pre_pad[vert_start:vert_end, horiz_start:horiz_end,:]
                    #单次卷积
                    Z[i,h,w,c] = conv_single_step(a_slice_pre, W[:,:,:,c], b[0,0,0,c])
    #检查数据格式
    assert(Z.shape == (m, n_h, n_w, n_c))
    #存储临时变量
    cache = (A_pre, W, b, hparameters)
    
    return (Z, cache)

def pool_forward(A_pre, hparameters, mode='max'):
    '''
    实现池化层的向前传播
    Parameters:
        A_pre - 输入数据，shape=(m, n_h_pre, n_w_pre, n_c_pre)
        hparameters - 包含超参数f和stride的字典
        mode - pooling模式选择（max,average）
    Returns:
        A - 池化层的输出，shape=(m, n_h, n_w, n_c)
        cache - 存储了反向传播需要用到的值，包含了参数和超参数
    '''
    # step1 获取输入数据的维度
    (m, n_h_pre, n_w_pre, n_c_pre) = A_pre.shape
    # step2 获取超参数
    f = hparameters['f']
    stride = hparameters['stride']
    # step3 计算输出维度
    n_h = np.floor((n_h - f) / stride) + 1
    n_w = np.floor((n_w - f) / stride) + 1
    n_c = n_c_pre
    
    # step4 初速化输出矩阵
    A = np.zeros((m, n_h, n_w, n_c)
    # step5 进行池化操作
    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                   vert_start = h * stride
                   vert_end = vert_start + f
                   horiz_start = w * stride
                   horiz_end = horiz_start + f
                   
                   a_slice_pre = A_pre[vert_start:vert_end, horiz_start:horiz_end, c]
                   
                   if mode ='max':
                       A[i, h, w, c] = np.max(a_slice_pre)
                    elif mode = 'average':
                        A[i, h, w, c] = np.mean(a_slice_pre)
    # 检查数据格式
    assert(A.shape = (m, n_h, n_w, n_c))
    
    # 存储临时变量
    cache = (A_pre, hparameters) 
    return A, cache



                    
                    
                    
                    
                    
if __name__ == '__main__':
    np.random.seed(1)    
    X = np.random.randn(4, 3, 3, 3)
    X_pad = zero_pad(X, pad=2)
    #绘图
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('X')
    ax[0].imshow(X[0,:,:,0])
    ax[1].set_title('X_pad')
    ax[1].imshow(X_pad[0,:,:,0])