# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:30:03 2018

@author: GEAR
"""
import numpy as np

def sigimoid(Z):
    
    A = 1 / (1 + np.exp(-Z))
    return A

def sigimoid_backward(dA, cache):
    
    Z = cache
    A = 1 / (1 + np.exp(-Z))
    dZ = dA * A * (1 - A)
    
    assert(dZ.shape == Z.shape)
    return dZ

def relu(Z):
    
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    return A

def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    
    return dZ