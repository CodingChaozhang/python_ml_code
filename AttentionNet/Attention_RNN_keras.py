# -*- coding: utf-8 -*-
"""
keras实现attention network

Created on Mon Dec 17 17:01:47 2018

@author: gear
"""
import keras.backend as K
from keras import layers
from keras import models
from keras import initializers  # In Keras 2.0, initializations was renamed (mirror) as initializers. 
# Attention GRU network
      
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]




MAX_SEQUENCE_LENGTH = 20

sequence_input = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = layers.Embedding(1000, 64)(sequence_input)
l_lstm = layers.Bidirectional(layers.LSTM(100, return_sequences=True))(embedded_sequences)
l_time= layers.TimeDistributed(layers.Dense(200))(l_lstm)
l_att = AttentionLayer()(l_time)
preds = layers.Dense(2, activation='softmax')(l_att)
model = models.Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['acc'])

print("model fitting - attention GRU network")
model.summary()
