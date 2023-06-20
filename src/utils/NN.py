import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Attention
from tensorflow.keras import regularizers

class Attention(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer='random_normal', trainable=True, name="attention_w")
        self.b = self.add_weight(shape=(input_shape[1],), initializer='zeros', trainable=True, name="attention_b")

    def call(self, inputs):
        e = tf.matmul(inputs, self.W) + self.b
        a = tf.nn.softmax(e, axis=1)
        weighted_sum = tf.reduce_sum(inputs * a, axis=1)
        return weighted_sum
    
#def get_model(input_shape: list):
#    model = Sequential()
#    model.add(LSTM(units=128, input_shape=input_shape, return_sequences=True, kernel_initializer='glorot_uniform'))
#    model.add(Attention())
#    #model.add(BatchNormalization())
#    model.add(Dense(units=32, activation='relu', kernel_initializer='glorot_uniform'))
#    model.add(Dense(units=1, activation='relu', kernel_initializer='glorot_uniform'))
#    return model


def get_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    lstm = tf.keras.layers.LSTM(units=input_shape[0], return_sequences=True)(inputs)
    attention = Attention(input_shape)(lstm)
    bn1 = BatchNormalization()(attention)
    dense = tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(bn1)
    bn2 = BatchNormalization()(dense)
    outputs = tf.keras.layers.Dense(units=1)(bn2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
