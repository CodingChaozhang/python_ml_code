# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io as sio
   
def load_2D_dataset(is_plot=True):
    data = sio.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    if is_plot:
        plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.reshape(211), s=40, cmap=plt.cm.Spectral);
    
    return train_X, train_Y, test_X, test_Y



