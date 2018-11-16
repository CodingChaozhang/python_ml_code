# -*- coding: utf-8 -*-
"""
简介：
实现目标检测领域的经典算法YOLO

Created on Wed Nov 14 20:14:29 2018

@author: dell
"""

import argparse
import os
import matplotlib.pyplot as plt  
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    '''
    通过阈值来过滤对象和分类的置信度
    Parameters:
        box_confidence - tensor类型，shape=(19,19,5,1),包含19x19单元格中每个单元格预测的5个锚框中的所有锚框的Pc(一些对象的置信概率)
        boxes - tensor类型， shape=(19,19,5,4), 包含了所有锚框的(bx, by, bh, bw)
        box_class_probs - tensor类型， shape=(19,19,5,80), 包含了所有单元格中所有对象(c1,c2,c3,...,c80)检测到的概率
        threshold - 阈值， 如果分类预测的概率高于它，那么这个分类预测的概率就会被保留
    Returns:
        scores - tensor类型， shape=(None, ), 包含了保留了锚框的分类概率
        boxes - tensor类型， shape=(None, 4), 包含了保留的锚框的(bx,by, bh, bw)
        classes - tensor类型， shape=(None, ), 包含了保留的锚框的索引
    '''
    
    # step1 计算锚框的得分
    box_scores = box_confidence * box_class_probs        # shape=(19,19,5,80)

    # step2 找到最大值的锚框的索引以及对于的最大值的锚框的分数
    box_classes = K.argmax(box_scores, axis=-1)         # shape=(19,19,5)
    box_class_scores = K.max(box_scores, axis=-1)       # shape=(19,19,5)
    
    # step3 根据阈值创建掩码
    filtering_mask = (box_class_scores >= threshold)
    
    # step4 对scores, boxes及classes使用掩码
    scores = tf.boolean_mask(box_class_scores, filtering_mask)      
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes

def iou(box1, box2):
    '''
    计算两个锚框的IOU(Intersction over Union)
    Parameters:
        box1 - 第一个锚框， tuple, (x1, y1, x2, y2)
        box2 - 第二个锚框， tuple, (x1, y1, x2, y2)
    Returns:
        iou - 实数， 两个锚框的iou值
    '''
    
    # step1 计算相交区域的坐标
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1]) 
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    
    # step2 计算交集的面积
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
    # step3 计算两个框的并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # step4 计算IOU
    iou = inter_area / union_area
    
    return iou

def lolo_non_max_supperession(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    '''
    为锚框实现非最大值抑制（Non-max suppression）
    Parameters:
        scores - tensor类型, shape=(None, ), yolo_filter_boxes()函数的输出
        boxes - tensor类型, shape(None, 4), yolo_filter_boxes()函数的输出
        classes - tensor类型, shape=(None, )
        max_boxes - 整数, 预测的锚框数量的最大值
        iou_threshold - 实数, 交并比(IOU)的阈值
    Returns:
        scores - 筛选后的值
        boxes - 筛选后的值
        classes - 筛选后的值
    '''
    
    max_boxes_tensor = K.variable(max_boxes, dtype=tf.int32)
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))   # 初始化max_boxes_tensor
    
    # 使用tf.image.non_max_suppression()来获取与我们保留的框相对饮的索引列表
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    
    # 使用K.gather()来选择保留的框
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes


    
        
    

    


    
    
    