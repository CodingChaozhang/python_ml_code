# -*- coding: utf-8 -*-
"""
简介：
用keras实现DeepDream
Created on Fri Nov 30 09:37:31 2018

@author: dell
"""
import scipy
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K

K.set_learning_phase(0)     # 禁用所有与训练有关的操作


# step1 加载预训练的Inception V3模型
# 构建不包括全连接层的Inception V3网络。使用预训练的ImageNet权重来加载模型
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

# step2 设置DeepDream配置
layer_contributions = {'mixed2':0.2,
                       'mixed3':3.0,
                       'mixed4':2.0,
                       'mixed5':3.0,
                       'mixed6':4.0}

# step3 定义需要最大化的损失
layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.0)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss += coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling
    
# step3 梯度上升过程
dream = model.input
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

# step5 定义辅助函数
def resize_img(img, size):
    img = np.copy(img)
    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
    
    return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)
    
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    
    return img

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.0
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x

# step4 在多个连续尺度上运行梯度上升
step = 0.01
num_octave = 3
octave_scale = 1.4
iterations = 20

max_loss = 10.0

# ----------------------------图像路径-------------------------------------#
base_image_path = 'H:/fish_video/fish_100.jpg'        

img = preprocess_image(base_image_path)
original_shape = img.shape[1:3]
sucessive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    sucessive_shapes.append(shape)
    
sucessive_shapes = sucessive_shapes[::-1]
original_img = np.copy(img)
shrunk_original_img = resize_img(img, sucessive_shapes[0])
for shape in sucessive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)
    
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img
    
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
    
save_img(img, fname='final_dream.png')



    


        


