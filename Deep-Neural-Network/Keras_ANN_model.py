# -*- coding: utf-8 -*-
"""
采用和Deep_ANN_model.py相同的数据训练keras模型
layers_dims = [12288, 20, 7, 5, 1]

Created on Sun Oct 28 19:31:58 2018

@author: GEAR
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import lr_utils  # 训练和测试数据

# step1 载入数据并做相应归一化
train_X, train_Y, test_X, test_Y, classes= lr_utils.load_dataset()
train_X = train_X.reshape(train_X.shape[0], -1) / 255
train_Y = train_Y.T
test_X = test_X.reshape(test_X.shape[0], -1) / 255
test_Y = test_Y.T

# step2 建立多层网络
model = Sequential()
model.add(Dense(units=20, input_shape=(12288,), activation='relu')) # 这里写成input_dims=12288也可以
model.add(Dense(units=7, activation='relu'))
model.add(Dense(units=5, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# step3 定义训练方式
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# step4 开始训练
model.fit(train_X, train_Y, epochs=300, batch_size=10)

# step5 评估模型
score = model.evaluate(test_X, test_Y)

# step6 进行预测
Y_prediction = model.predict_classes(test_X)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


