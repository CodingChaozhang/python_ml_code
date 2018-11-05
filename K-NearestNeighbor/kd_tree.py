# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:57:30 2018

@author: GEAR
"""

'''
kdTree构建
'''
import sys
reload(sys)

# kd树的每个结点中主要包含的数据如下
class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt # k维空间的一个样本点
        self.split = split     # 进行分割维度的序号
        self.left = left
        self.right = right
        
class kdTree(object):
    def __init__(self, dataSet):
        k = dataSet.shape[1]  # 数据的维度
        
        #按第split维划分数据集
        def creatNode(split, dataSet):
            if dataSet is None:
                return None
            # 
        
        